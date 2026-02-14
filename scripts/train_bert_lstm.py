import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

class EssayDataset(Dataset):
    def __init__(self, essays, scores, nlp_features=None, tokenizer=None, max_len=512):
        self.essays = essays
        self.scores = scores
        self.nlp_features = nlp_features
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.essays)
    
    def __getitem__(self, idx):
        essay = str(self.essays[idx])
        score = self.scores[idx]
        
        encoding = self.tokenizer.encode_plus(
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        item = {
            'essay_text': essay,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(score, dtype=torch.float)
        }
        
        if self.nlp_features is not None:
            item['nlp_features'] = torch.tensor(self.nlp_features[idx], dtype=torch.float)
            
        return item

class BertLstmModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=256, num_layers=1, output_dim=1, nlp_dim=0, dropout=0.3):
        super(BertLstmModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        combined_dim = (hidden_dim * 2) + nlp_dim
        
        self.fusion = nn.Linear(combined_dim, 128)
        self.relu = nn.ReLU()
        
        self.fc = nn.Linear(128, output_dim)
        
    def forward(self, input_ids, attention_mask, nlp_features=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(sequence_output)
        
        # lstm_out: [Batch, Seq, Dim] -> [Batch, Dim, Seq]
        lstm_out_permuted = lstm_out.permute(0, 2, 1)
        pooled = torch.nn.functional.max_pool1d(lstm_out_permuted, kernel_size=lstm_out_permuted.shape[2]).squeeze(2)
        
        if nlp_features is not None:
            combined = torch.cat((pooled, nlp_features), dim=1)
        else:
            combined = pooled
        
        fused = self.relu(self.fusion(combined))
        out = self.dropout(fused)
        return self.fc(out)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    total_loss = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        nlp_features = d.get("nlp_features", None)
        if nlp_features is not None:
            nlp_features = nlp_features.to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nlp_features=nlp_features
        )
        
        loss = loss_fn(outputs, targets)
        
        total_loss += loss.item() * input_ids.size(0)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return total_loss / len(data_loader.dataset)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    total_loss = 0
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            nlp_features = d.get("nlp_features", None)
            if nlp_features is not None:
                nlp_features = nlp_features.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                nlp_features=nlp_features
            )
            
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * input_ids.size(0)
            
            predictions.extend(outputs.cpu().detach().numpy())
            real_values.extend(targets.cpu().detach().numpy())
            
    return total_loss / len(data_loader.dataset), np.array(predictions), np.array(real_values)


def run_train(model, train_loader, val_loader, epochs, lr, device, target_cols):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        val_loss, preds, true_vals = eval_model(model, val_loader, loss_fn, device)
        
        preds_int = np.clip(np.round(preds), 0, 4).astype(int)
        true_int = np.round(true_vals).astype(int)
        overall_qwk = np.mean([cohen_kappa_score(true_int[:, i], preds_int[:, i], weights='quadratic') for i in range(len(target_cols))])
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | QWK: {overall_qwk:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    return best_model_state, best_val_loss

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="End-to-End BERT+LSTM Training for AES")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--essays", type=str, default=r"data\asap-set-7-essays.csv")
    parser.add_argument("--scores", type=str, default=r"data\asap-set-7-scores.csv")
    parser.add_argument("--nlp", type=str, default=r"data\hand-crafted-features.csv")
    parser.add_argument("--output", type=str, default=r"outputs\lstm_predictions.csv")
    parser.add_argument("--model_save", type=str, default=r"models\final_lstm_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--folds", type=int, default=5)
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading data...")
    essays_df = pd.read_csv(args.essays, encoding='ISO-8859-1')
    scores_df = pd.read_csv(args.scores)
    nlp_df = pd.read_csv(args.nlp)
    
    id_col = "essay_id"
    essay_col = "essay"
    score_cols = ["ideas", "organization", "style", "conventions"]
    rater_col = "rater"
    
    pivoted = scores_df.pivot(index=id_col, columns=rater_col, values=score_cols)
    pivoted.columns = [f"R{rater}_{score}" for score, rater in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    data = pd.merge(essays_df, pivoted, on=id_col)
    data = pd.merge(data, nlp_df, on=id_col)
    
    nlp_cols = [c for c in nlp_df.columns if c != id_col]
    nlp_features_data = StandardScaler().fit_transform(data[nlp_cols].values)
    nlp_dim = nlp_features_data.shape[1]
    
    target_cols = [f"R{r}_{s}" for r in [1, 2] for s in score_cols]
    X_text = data[essay_col].values
    Y_scores = data[target_cols].values
    essay_ids = data[id_col].values
    
    y_mean = Y_scores.mean(axis=1)
    y_binned = np.round(y_mean).astype(int)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    all_results = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_text, y_binned), 1):
        print(f"\n--- Fold {fold}/{args.folds} ---")
        train_dataset = EssayDataset(X_text[train_index], Y_scores[train_index], nlp_features_data[train_index], tokenizer)
        val_dataset = EssayDataset(X_text[val_index], Y_scores[val_index], nlp_features_data[val_index], tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        model = BertLstmModel(output_dim=len(target_cols), nlp_dim=nlp_dim).to(device)
        best_state, _ = run_train(model, train_loader, val_loader, args.epochs, args.lr, device, target_cols)
        
        model.load_state_dict(best_state)
        _, preds, _ = eval_model(model, val_loader, nn.MSELoss(), device)
        preds = np.clip(np.round(preds), 0, 4).astype(int)
        
        fold_ids = essay_ids[val_index]
        fold_true = Y_scores[val_index]
        
        for i, eid in enumerate(fold_ids):
            # Rater 1
            all_results.append({
                "essay_id": eid,
                "fold": fold,
                "rater": 1,
                "true_ideas": fold_true[i, 0],
                "pred_ideas": preds[i, 0],
                "true_organization": fold_true[i, 1],
                "pred_organization": preds[i, 1],
                "true_style": fold_true[i, 2],
                "pred_style": preds[i, 2],
                "true_conventions": fold_true[i, 3],
                "pred_conventions": preds[i, 3]
            })
            # Rater 2
            all_results.append({
                "essay_id": eid,
                "fold": fold,
                "rater": 2,
                "true_ideas": fold_true[i, 4],
                "pred_ideas": preds[i, 4],
                "true_organization": fold_true[i, 5],
                "pred_organization": preds[i, 5],
                "true_style": fold_true[i, 6],
                "pred_style": preds[i, 6],
                "true_conventions": fold_true[i, 7],
                "pred_conventions": preds[i, 7]
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values(by=['rater', 'essay_id'])
    df_results.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    
    print(f"\n--- Training Final Model on Full Dataset ---")
    full_dataset = EssayDataset(X_text, Y_scores, nlp_features_data, tokenizer)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
    
    final_model = BertLstmModel(output_dim=len(target_cols), nlp_dim=nlp_dim).to(device)
    optimizer = AdamW(final_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    loss_fn = nn.MSELoss()
    
    for epoch in range(args.epochs):
        loss = train_epoch(final_model, full_loader, loss_fn, optimizer, device, scheduler)
        print(f"Final Model Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")
        
    os.makedirs(os.path.dirname(args.model_save), exist_ok=True)
    torch.save(final_model.state_dict(), args.model_save)
    print(f"Final model saved to {args.model_save}")

if __name__ == "__main__":
    main()
