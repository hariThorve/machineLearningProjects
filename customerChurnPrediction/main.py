import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

columns_to_drop = [
    'customerID',  
    'gender',      
    'PhoneService', 
    'MultipleLines',
    'StreamingTV', 
    'StreamingMovies' 
]


df_cleaned = df.drop(columns=columns_to_drop)

missing_values = df_cleaned.isnull().sum()


# Convert 'TotalCharges' to numeric, forcing errors to NaN
df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
# Example strategy: Fill missing values for 'TotalCharges' with the median
df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].fillna(df_cleaned['TotalCharges'].median())


missing_values_after = df_cleaned.isnull().sum()
# Identify categorical columns to encode
categorical_columns = [
    'Partner', 
    'Dependents', 
    'InternetService', 
    'OnlineSecurity', 
    'OnlineBackup', 
    'DeviceProtection', 
    'TechSupport', 
    'Contract', 
    'PaperlessBilling', 
    'PaymentMethod'
]

# Apply one-hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_columns, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop(['Churn'], axis=1)  # Features
y = df_encoded['Churn'].map({'Yes': 1, 'No': 0})  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Choose a Model
model = RandomForestClassifier(random_state=42)

# 2. Train the Model
model.fit(X_train, y_train)

# 3. Make Predictions
y_pred = model.predict(X_test)

# 4. Evaluate the Model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Check the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the model: {accuracy:.2f}")

model_score = model.score(X_test, y_test)
print(f"Model score (mean accuracy): {model_score:.2f}")
