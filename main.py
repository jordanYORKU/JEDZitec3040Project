import csv
from collections import Counter
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def preprocess_data(data):
    """Preprocess data before building tree"""
    processed = []
    
    for row in data:
        new_row = row.copy()
        
        # Bin ages
        if row[1]:
            age = int(float(row[1]))
            if age < 20:
                new_row[1] = "18-19"
            elif age < 23:
                new_row[1] = "20-22"
            else:
                new_row[1] = "23+"
        else:
            new_row[1] = "21-22"  
        
        # Handle missing Gender
        if not row[2]:
            new_row[2] = "Female"
        
        # Group ticket types
        ticket = row[4]
        if ticket in ["Jedz Team", "Jedz Boyz", "Jedz Guest", "Photographer Entry", "VIP"]:
            new_row[4] = "Staff_Free"
        elif ticket == "Photographer Plus One Entry":
            new_row[4] = "Friend"
        elif "Early Bird" in ticket:
            new_row[4] = "Early"
        elif "Secret" in ticket:
            new_row[4] = "Secret"
        elif ticket in ["General Admission", "Extended Admission", "Late Admission"]:
            new_row[4] = "Regular_Late"
        elif ticket == "Final Admission":
            new_row[4] = "Last_Minute"
        else:
            new_row[4] = "Other"
        
        # Bin prices into numeric tiers (0-9, 10-19, 20-29, 30+)
        if row[5]:
            price = float(row[5])
            if price < 10:
                new_row[5] = 0
            elif price < 20:
                new_row[5] = 1
            elif price < 30:
                new_row[5] = 2
            else:
                new_row[5] = 3
        else:
            new_row[5] = None  # Will be imputed later
        
        processed.append(new_row)
    
    return processed

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def shannon_entropy(self, labels):
        if not labels:
            return 0
        counts = Counter(labels)
        total = len(labels)
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
    
    def weighted_entropy(self, data, feature, target_col):
        total = len([row for row in data if row[target_col]])
        weighted_ent = 0
        
        values = set(row[feature] for row in data if row[feature] is not None)
        for value in values:
            subset_labels = [row[target_col] for row in data 
                           if row[feature] == value and row[target_col]]
            weight = len(subset_labels) / total
            weighted_ent += weight * self.shannon_entropy(subset_labels)
        
        return weighted_ent
    
    def fit(self, data, target_col):
        self.tree = self._build(data, target_col, 0)
    
    def _build(self, data, target_col, depth):
        labels = [row[target_col] for row in data if row[target_col]]
        
        if not labels:
            return "Unknown"
        
        current_entropy = self.shannon_entropy(labels)
        label_counts = Counter(labels)
        
        if current_entropy == 0 or depth >= self.max_depth:
            result = label_counts.most_common(1)[0][0]
            return result
        
        features = [i for i in range(len(data[0])) if i != target_col and i != 0]
        
        entropies = {}
        for feature in features:
            ent = self.weighted_entropy(data, feature, target_col)
            entropies[feature] = ent
        
        best_feature = min(entropies, key=entropies.get)
        
        tree = {'feature': best_feature, 'branches': {}}
        values = set(row[best_feature] for row in data if row[best_feature] is not None)
        
        for value in values:
            subset = [row for row in data if row[best_feature] == value]
            tree['branches'][value] = self._build(subset, target_col, depth + 1)
        
        return tree
    
    def predict_one(self, row):
        """Predict for a single row"""
        node = self.tree
        while isinstance(node, dict):
            feature = node['feature']
            value = row[feature]
            if value in node['branches']:
                node = node['branches'][value]
            else:
                # If value not seen in training, return most common
                return "Yes"  # Default prediction
        return node
    
    def predict(self, data):
        """Predict for multiple rows"""
        return [self.predict_one(row) for row in data]
    
    def display(self, node=None, indent=0):
        if node is None:
            node = self.tree
        
        if isinstance(node, dict):
            feature_names = ['Index', 'Age', 'Gender', 'Checked In', 'Ticket Type', 'Buyer Pays', 'Name Origin']
            print('  ' * indent + f"{feature_names[node['feature']]}:")
            for value, subtree in node['branches'].items():
                print('  ' * (indent + 1) + f"-> {value}")
                self.display(subtree, indent + 2)
        else:
            print('  ' * indent + f"Predict: {node}")
    
    def get_depth(self, node=None):
        if node is None:
            node = self.tree
        
        if isinstance(node, dict):
            return 1 + max(self.get_depth(subtree) for subtree in node['branches'].values())
        return 0

# Load data
data = []
with open('partydata.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        data.append(row)

print(f"Loaded {len(data)} rows")

# Preprocess ALL data first
data = preprocess_data(data)
print("Data preprocessed")

# === SPLIT DATA FIRST (80/20) ===
print("\n" + "="*60)
print("SPLITTING DATA")
print("="*60)

# Convert to DataFrame for easy splitting
df = pd.DataFrame(data, columns=header)
df["Attended"] = (df["Checked In"] == "Yes").astype(int)

# Convert Buyer Pays to numeric
df["Buyer Pays"] = pd.to_numeric(df["Buyer Pays"], errors='coerce')

# Split 80/20 with same random state
train_indices, test_indices = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df["Attended"]
)

# Split the list data for Decision Tree
data_train = [data[i] for i in train_indices]
data_test = [data[i] for i in test_indices]

# Split DataFrame for other models
df_train = df.loc[train_indices]
df_test = df.loc[test_indices]

print(f"Training set: {len(data_train)} samples")
print(f"Test set: {len(data_test)} samples")

# Save test set with actual check-in status
test_set_output = df_test[["Age", "Gender", "Ticket Type", "Buyer Pays", "Predicted Name Origin", "Checked In", "Attended"]]
test_set_output.to_csv('test_set_with_actual.csv', index=True)
print(f"Saved test set to: test_set_with_actual.csv")

# === DECISION TREE ===
print("\n" + "="*60)
print("DECISION TREE")
print("="*60)

tree = DecisionTree(max_depth=3)
tree.fit(data_train, target_col=3)
tree.display()
print("="*60)
print(f"Tree depth: {tree.get_depth()}")

# Predict on test set
y_pred_dt = tree.predict(data_test)
y_actual_dt = [row[3] for row in data_test]

# Calculate accuracy
correct_dt = sum(1 for pred, actual in zip(y_pred_dt, y_actual_dt) if pred == actual and actual)
total_dt = sum(1 for actual in y_actual_dt if actual)
accuracy_dt = correct_dt / total_dt if total_dt > 0 else 0

print(f"\nTest Accuracy: {accuracy_dt*100:.1f}%")

# Save predictions
dt_results = pd.DataFrame({
    'Index': test_indices,
    'Actual': y_actual_dt,
    'Predicted': y_pred_dt,
    'Correct': [pred == actual for pred, actual in zip(y_pred_dt, y_actual_dt)]
})
dt_results.to_csv('decision_tree_predictions.csv', index=False)
print("Saved predictions to: decision_tree_predictions.csv")

# === NAIVE BAYES ===
print("\n" + "="*60)
print("NAIVE BAYES")
print("="*60)

# Prepare data
X_train = df_train[["Age", "Gender", "Ticket Type", "Buyer Pays", "Predicted Name Origin"]]
y_train = df_train["Attended"]
X_test = df_test[["Age", "Gender", "Ticket Type", "Buyer Pays", "Predicted Name Origin"]]
y_test = df_test["Attended"]

# Preprocessor - Buyer Pays is now numeric (0, 1, 2, 3)
numeric_features = ["Buyer Pays"]
categorical_features = ["Age", "Gender", "Ticket Type", "Predicted Name Origin"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Model
nb_model = Pipeline(steps=[
    ("prepr", preprocessor),
    ("to_dense", FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else x)),
    ("clf", GaussianNB()),
])

# Train
nb_model.fit(X_train, y_train)

# Predict
y_pred_nb = nb_model.predict(X_test)
y_pred_nb_train = nb_model.predict(X_train)

# Evaluate
accuracy_nb = accuracy_score(y_test, y_pred_nb)
train_accuracy_nb = accuracy_score(y_train, y_pred_nb_train)

print(f"Training Accuracy: {train_accuracy_nb*100:.1f}%")
print(f"Test Accuracy: {accuracy_nb*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

# Save predictions
nb_results = pd.DataFrame({
    'Index': test_indices,
    'Actual': y_test.values,
    'Predicted': y_pred_nb,
    'Correct': (y_test.values == y_pred_nb)
})
nb_results.to_csv('naive_bayes_predictions.csv', index=False)
print("Saved predictions to: naive_bayes_predictions.csv")

# === LOGISTIC REGRESSION ===
print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

# Model (using same preprocessor)
lr_model = Pipeline(steps=[
    ("prepr", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

# Train
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)
y_pred_lr_train = lr_model.predict(X_train)

# Evaluate
accuracy_lr = accuracy_score(y_test, y_pred_lr)
train_accuracy_lr = accuracy_score(y_train, y_pred_lr_train)

print(f"Training Accuracy: {train_accuracy_lr*100:.1f}%")
print(f"Test Accuracy: {accuracy_lr*100:.1f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Save predictions
lr_results = pd.DataFrame({
    'Index': test_indices,
    'Actual': y_test.values,
    'Predicted': y_pred_lr,
    'Correct': (y_test.values == y_pred_lr)
})
lr_results.to_csv('logistic_regression_predictions.csv', index=False)
print("Saved predictions to: logistic_regression_predictions.csv")

# === SUMMARY ===
print("\n" + "="*60)
print("ALL 3 MODELS COMPLETE!")
print("="*60)
print(f"Decision Tree Test Accuracy: {accuracy_dt*100:.1f}%")
print(f"Naive Bayes Test Accuracy: {accuracy_nb*100:.1f}%")
print(f"Logistic Regression Test Accuracy: {accuracy_lr*100:.1f}%")
print("\nPrice Tiers Used:")
print("  Tier 0: $0-9.99")
print("  Tier 1: $10-19.99")
print("  Tier 2: $20-29.99")
print("  Tier 3: $30+")
print("\nFiles saved:")
print("  1. test_set_with_actual.csv - The 20% test data with actual check-in")
print("  2. decision_tree_predictions.csv")
print("  3. naive_bayes_predictions.csv")
print("  4. logistic_regression_predictions.csv")
    
