import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, cohen_kappa_score

class NeuralDataset(Dataset):
    def __init__(self, bert_embeddings, nlp_features, targets):
        self.bert_embeddings = torch.FloatTensor(bert_embeddings)
        self.nlp_features = torch.FloatTensor(nlp_features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.bert_embeddings[idx], self.nlp_features[idx], self.targets[idx]

class SplitBranchFNN(nn.Module):
    """
    Split-Branch MLP architecture:
    - Branch A: Semantic (BERT 768 -> 256 -> 128)
    - Branch B: Stylistic (NLP Dim -> 64 -> 32)
    - Fusion: (160 -> 64 -> 8)
    """
    def __init__(self, nlp_dim, output_dim=8, dropout=0.3):
        super(SplitBranchFNN, self).__init__()
        
        # Branch A: BERT features
        self.bert_branch = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Branch B: NLP features
        self.nlp_branch = nn.Sequential(
            nn.Linear(nlp_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, x_bert, x_nlp):
        out_bert = self.bert_branch(x_bert)
        out_nlp = self.nlp_branch(x_nlp)
        
        combined = torch.cat([out_bert, out_nlp], dim=1)
        fused = self.fusion(combined)
        return self.output_layer(fused)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_bert, x_nlp, y in dataloader:
        x_bert, x_nlp, y = x_bert.to(device), x_nlp.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_bert, x_nlp)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_bert.size(0)
    return total_loss / len(dataloader.dataset)

def eval_model(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_bert, x_nlp, _ in dataloader:
            x_bert, x_nlp = x_bert.to(device), x_nlp.to(device)
            outputs = model(x_bert, x_nlp)
            predictions.append(outputs.cpu().numpy())
    return np.vstack(predictions)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Split-Branch FNN Training for AES")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--essays", type=str, default=r"data\asap-set-7-essays.csv")
    parser.add_argument("--scores", type=str, default=r"data\asap-set-7-scores.csv")
    parser.add_argument("--bert", type=str, default=r"data\bert-base-uncased.csv")
    parser.add_argument("--nlp", type=str, default=r"data\hand-crafted-features.csv")
    parser.add_argument("--output", type=str, default=r"outputs\fnn_predictions.csv")
    parser.add_argument("--model_save", type=str, default=r"models\final_fnn_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    scores_df = pd.read_csv(args.scores)
    bert_df = pd.read_csv(args.bert)
    nlp_df = pd.read_csv(args.nlp)

    id_col = "essay_id"
    score_cols = ["ideas", "organization", "style", "conventions"]
    rater_col = "rater"

    pivoted = scores_df.pivot(index=id_col, columns=rater_col, values=score_cols)
    pivoted.columns = [f"R{rater}_{score}" for score, rater in pivoted.columns]
    pivoted = pivoted.reset_index()

    data = pivoted.merge(bert_df, on=id_col).merge(nlp_df, on=id_col)
    
    bert_cols = [c for c in bert_df.columns if c != id_col]
    nlp_cols = [c for c in nlp_df.columns if c != id_col]
    target_cols = [f"R{r}_{s}" for r in [1, 2] for s in score_cols]

    X_bert = data[bert_cols].values
    X_nlp = StandardScaler().fit_transform(data[nlp_cols].values)
    Y = data[target_cols].values
    essay_ids = data[id_col].values

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    y_mean_binned = np.round(Y.mean(axis=1)).astype(int)
    
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_bert, y_mean_binned), 1):
        print(f"\n--- Fold {fold}/{args.folds} ---")
        train_ds = NeuralDataset(X_bert[train_idx], X_nlp[train_idx], Y[train_idx])
        val_ds = NeuralDataset(X_bert[val_idx], X_nlp[val_idx], Y[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        
        model = SplitBranchFNN(nlp_dim=len(nlp_cols)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, xn, y in val_loader:
                    xb, xn, y = xb.to(device), xn.to(device), y.to(device)
                    val_loss += criterion(model(xb, xn), y).item() * xb.size(0)
            val_loss /= len(val_ds)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        model.load_state_dict(best_state)
        preds = eval_model(model, val_loader, device)
        preds = np.clip(np.round(preds), 0, 4).astype(int)
        
        fold_ids = essay_ids[val_idx]
        fold_true = Y[val_idx]
        
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

    print("\n--- Training Final Model on Full Dataset ---")
    full_ds = NeuralDataset(X_bert, X_nlp, Y)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)
    
    final_model = SplitBranchFNN(nlp_dim=len(nlp_cols)).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        loss = train_epoch(final_model, full_loader, optimizer, nn.MSELoss(), device)
        if (epoch + 1) % 10 == 0:
            print(f"Final Model Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")
            
    os.makedirs(os.path.dirname(args.model_save), exist_ok=True)
    torch.save(final_model.state_dict(), args.model_save)
    print(f"Final model saved to {args.model_save}")

if __name__ == "__main__":
    main()