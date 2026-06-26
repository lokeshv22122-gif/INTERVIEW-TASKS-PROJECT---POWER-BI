# Loan Delinquency Streamlit App

This project deploys your **KNN model** for predicting **Serious Delinquency (`Sdelinquent`)** using Streamlit.

## Files
- `train_model.py` → trains the model and saves artifacts
- `app.py` → Streamlit web app
- `requirements.txt` → dependencies
- `Loan Delinquent Dataset.csv` → dataset

## Project structure
```bash
loan_delinquent_app/
│
├── Loan Delinquent Dataset.csv
├── train_model.py
├── app.py
├── requirements.txt
├── knn_model.pkl
├── scaler.pkl
├── label_encoders.pkl
└── feature_columns.pkl
```

## Step 1: Create virtual environment (recommended)
```bash
python -m venv venv
```

### Activate on Windows
```bash
venv\Scripts\activate
```

## Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Train the model
```bash
python train_model.py
```

This will create:
- `knn_model.pkl`
- `scaler.pkl`
- `label_encoders.pkl`
- `feature_columns.pkl`

## Step 4: Run Streamlit app
```bash
streamlit run app.py
```

## Model details
- Algorithm: **KNN Classifier**
- Target column: **Sdelinquent**
- Features used:
  - `term`
  - `gender`
  - `purpose`
  - `home_ownership`
  - `age`
  - `FICO`

## Notes
- The deployment uses **StandardScaler** before KNN training/prediction.
- The script performs **GridSearchCV** to automatically pick the best `k`.
- Based on your notebook, the best `k` found was **11**.
