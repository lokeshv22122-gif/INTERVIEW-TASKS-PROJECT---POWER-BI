import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("Loan Delinquent Dataset.csv")

# =========================
# 2. Select only the same features used in Streamlit app
# =========================
feature_cols = ['term', 'gender', 'purpose', 'home_ownership', 'age', 'FICO']
target_col = 'Sdelinquent'

X = df[feature_cols].copy()
y = df[target_col]

# =========================
# 3. Encode ALL selected columns explicitly
#    (force convert to string first)
# =========================
label_encoders = {}

for col in feature_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# =========================
# 4. Scale features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. Train-test split with stratify
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. Tune K using GridSearchCV
# =========================
param_grid = {"n_neighbors": list(range(1, 16))}

grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='f1'
)

grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_

# =========================
# 7. Evaluate model
# =========================
y_pred = model.predict(X_test)

print("Best K:", grid.best_params_["n_neighbors"])
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nTarget Distribution:")
print(df[target_col].value_counts())
print("\nTarget Distribution %:")
print(df[target_col].value_counts(normalize=True) * 100)

# =========================
# 8. Save artifacts
# =========================
joblib.dump(model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")

print("\nArtifacts saved successfully:")
print("- knn_model.pkl")
print("- scaler.pkl")
print("- label_encoders.pkl")
print("- feature_columns.pkl")