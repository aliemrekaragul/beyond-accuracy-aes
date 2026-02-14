from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from tqdm import tqdm
import os

def get_bert_embeddings(essay_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokens = tokenizer(essay_text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**tokens)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

import argparse

def run(input_path, output_path=None):
    if output_path is None:
        output_path = os.path.dirname(input_path)
    
    id_col_name = "essay_id"
    essay_col_name = "essay"
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, encoding='ISO-8859-1')
    essays = df[essay_col_name]
    essay_ids = df[id_col_name]

    tqdm.pandas()
    print("Extracting BERT embeddings...")
    bert_embeddings = essays.progress_apply(get_bert_embeddings)
    bert_embeddings_df = pd.DataFrame(bert_embeddings.tolist())
    bert_embeddings_df.insert(0, id_col_name, essay_ids)

    output_file = os.path.join(output_path, "bert-base-uncased.csv")
    bert_embeddings_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract BERT embeddings from essays.")
    parser.add_argument("--input", default=r"data\asap-set-7-essays.csv", help="Path to the input CSV file.")
    parser.add_argument("--output", help="Directory to save the output CSV. Defaults to input directory.")
    
    args = parser.parse_args()
    run(args.input, args.output)