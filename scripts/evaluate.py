import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import cohen_kappa_score
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

def calculate_qwk(y_true, y_pred):
    """Calculates Quadratic Weighted Kappa."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def process_predictions(file_path):
    """Loads prediction CSV and calculates QWK for each domain and rater."""
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    traits = ['ideas', 'organization', 'style', 'conventions']
    raters = [1, 2]
    
    results = {}

    for trait in traits:
        trait_qwks = []
        for rater in raters:
            rater_df = df[df['rater'] == rater]
            if rater_df.empty:
                continue
                
            true_col = f"true_{trait}"
            pred_col = f"pred_{trait}"
            
            if true_col in rater_df.columns and pred_col in rater_df.columns:
                qwk = calculate_qwk(rater_df[true_col], rater_df[pred_col])
                results[f"R{rater}_{trait}"] = qwk
                trait_qwks.append(qwk)
        
        if trait_qwks:
            results[f"Avg_{trait}"] = np.mean(trait_qwks)

    for rater in raters:
        rater_values = [results[f"R{rater}_{trait}"] for trait in traits if f"R{rater}_{trait}" in results]
        if rater_values:
            results[f"Overall_R{rater}"] = np.mean(rater_values)
            
    avg_trait_values = [results[f"Avg_{trait}"] for trait in traits if f"Avg_{trait}" in results]
    if avg_trait_values:
        results["Overall_Average"] = np.mean(avg_trait_values)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate AES models (LSTM and FNN)")
    parser.add_argument("--lstm_preds", type=str, default=r"outputs\lstm_predictions.csv", help="Path to LSTM predictions CSV")
    parser.add_argument("--fnn_preds", type=str, default=r"outputs\fnn_predictions.csv", help="Path to FNN predictions CSV")
    parser.add_argument("--output", type=str, default=r"outputs\evaluation_summary.csv", help="Path to save summary CSV")
    
    args = parser.parse_args()
    
    models = {
        "LSTM": args.lstm_preds,
        "FNN": args.fnn_preds
    }
    
    summary_data = []
    
    traits = ['ideas', 'organization', 'style', 'conventions']
    
    for model_name, path in models.items():
        print(f"Evaluating {model_name} from {path}...")
        results = process_predictions(path)
        if results:
            row_avg = [f"{model_name} (Avg)"]
            for trait in traits:
                row_avg.append(f"{results.get(f'Avg_{trait}', 0):.4f}")
            row_avg.append(f"{results.get('Overall_Average', 0):.4f}")
            summary_data.append(row_avg)
            
            row_r1 = [f"{model_name} (R1)"]
            for trait in traits:
                row_r1.append(f"{results.get(f'R1_{trait}', 0):.4f}")
            row_r1.append(f"{results.get('Overall_R1', 0):.4f}")
            summary_data.append(row_r1)
            
            row_r2 = [f"{model_name} (R2)"]
            for trait in traits:
                row_r2.append(f"{results.get(f'R2_{trait}', 0):.4f}")
            row_r2.append(f"{results.get('Overall_R2', 0):.4f}")
            summary_data.append(row_r2)
            
            summary_data.append(["-" * 12] * 6) 
        else:
            print(f"Warning: Could not process {model_name} results at {path}")
            
    if summary_data:
        headers = ["Model/Rater", "Ideas", "Org", "Style", "Conv", "Overall QWK"]
        
        print("\nEvaluation Results (Quadratic Weighted Kappa):")
        if tabulate:
            print(tabulate(summary_data, headers=headers, tablefmt="grid"))
        else:
            print(" | ".join(headers))
            print("-" * 80)
            for row in summary_data:
                print(" | ".join(map(str, row)))
        
        save_data = [r for r in summary_data if r[0] != "-" * 12]
        summary_df = pd.DataFrame(save_data, columns=headers)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        summary_df.to_csv(args.output, index=False)
        print(f"\nDetailed summary saved to: {args.output}")
    else:
        print("\nError: No valid prediction files were found. Please ensure training has completed successfully.")

if __name__ == "__main__":
    main()