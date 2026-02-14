# Beyond Accuracy: Using Multi-facet Rasch Measurement Model to Validate Automated Essay Scoring in Analytical Writing Assessment
This is an implementation of the paper "Beyond Accuracy: Using Multi-facet Rasch Measurement Model to Validate Automated Essay Scoring in Analytical Writing Assessment".

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18641673.svg)](https://doi.org/10.5281/zenodo.18641673)
## Setup and Installation

1. **Clone the repository**.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Run the scripts**:
You can run all the scripts with default parameters to reproduce the results in the paper. Run the scripts in the same order as in the Usage section.

To manipulate the parameters, see the help message for each script by running `python scripts/<script_name>.py --help`.

In order to obtain the evaluation results, you need to run `evaluate.py`. It will save the evaluation results (QWK stats) in `outputs/evaluation_summary.csv`.

## Usage
The scripts should be run in the same order because the training scripts require the output of the feature extraction scripts.

### To calculate hand crafted features with default parameters run:

```
python scripts/vec_nlp.py
```
This will save the features in `data/hand-crafted-features.csv`
It will also save the scaler in `models/hand_craft_scaler.pkl`

### To calculate BERT vectors with default parameters run:
```
python scripts/vec_bert.py
```
This will save the vectors in `data/bert-base-uncased.csv`

### To train FNN model with default parameters run:
```
python scripts/train_fnn.py
```
This will save the model in `models/final_fnn_model.pt`
Also, the predictions will be saved into `ouptuts/fnn_predictions.pt`

### To train BERT+LSTM model with default parameters run:
```
python scripts/train_bert_lstm.py
```
This will save the model in `models/final_lstm_model.pt`
Also, the predictions will be saved into `ouptuts/lstm_predictions.pt`

### To evaluate the models with default parameters run:
```
python scripts/evaluate.py
```
This will save the evaluation results (QWK stats) in `outputs/evaluation_summary.csv`

## Multi-facet Rasch Model Analysis
`mfrm` directory contains the FACETS model definitions. You can simply drag and drop the `.txt` files into the FACETS software to run the analysis. 
The free version of Facets (MINIFAC) will not be able to run the analysis with the complete datasets. The full version of Facets with license can be obtained from https://www.winsteps.com/facets.htm 