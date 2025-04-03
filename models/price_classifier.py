import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set up paths
ROOT = Path(__file__).parent.parent
CLEAN_DATASET = ROOT / 'data' / 'processed' / 'clean_dataset.parquet'

# Load data
df = pd.read_parquet(CLEAN_DATASET, engine='pyarrow')

# Print initial dataset size
print(f"\nInitial dataset size: {len(df)}")

# Create price categories
price_bins = [0, 75000, 150000, 300000, float('inf')]
price_labels = ['Low', 'Medium', 'High', 'Luxury']
df['price_category'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)

# Select features
features = [
    'build_area',
    'construction_year',
    'floor_number',
    'latitude',
    'longitude',
    'number_of_bedrooms',
    'renovated'
]

# Print feature info before preprocessing
print("\nFeature information before preprocessing:")
for feature in features:
    print(f"\n{feature}:")
    print(f"Unique values: {df[feature].nunique()}")
    print(f"Missing values: {df[feature].isna().sum()}")
    print(f"Data type: {df[feature].dtype}")

# Prepare X and y
X = df[features].copy()
y = df['price_category']

# Remove rows where price_category is NaN (if any)
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]

print(f"\nDataset size after removing NaN price categories: {len(X)}")

# Handle missing values in features
X = X.fillna({
    'construction_year': X['construction_year'].median(),
    'floor_number': X['floor_number'].median(),
    'number_of_bedrooms': X['number_of_bedrooms'].median(),
    'renovated': 0  # Assuming 0 means not renovated
})

# Convert renovated to int if it's not already
X['renovated'] = X['renovated'].fillna(0).astype(int)

# Print feature info after preprocessing
print("\nFeature information after preprocessing:")
for feature in features:
    print(f"\n{feature}:")
    print(f"Unique values: {X[feature].nunique()}")
    print(f"Missing values: {X[feature].isna().sum()}")
    print(f"Data type: {X[feature].dtype}")

# Print class distribution
print("\nClass distribution:")
print(y.value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Add class weights to handle imbalance
)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Price Classification')
plt.tight_layout()
plt.savefig(ROOT / 'visualizations' / 'feature_importance.png')

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=price_labels,
            yticklabels=price_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(ROOT / 'visualizations' / 'confusion_matrix.png')

# Save the model and scaler
import joblib
model_path = ROOT / 'models' / 'random_forest_price_classifier.joblib'
scaler_path = ROOT / 'models' / 'price_classifier_scaler.joblib'
joblib.dump(rf_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nModel saved to {model_path}")
print(f"Scaler saved to {scaler_path}")

# Example prediction function
def predict_price_category(features_dict):
    """
    Predict price category for new data.
    
    Args:
        features_dict (dict): Dictionary with features:
            - build_area
            - construction_year
            - floor_number
            - latitude
            - longitude
            - number_of_bedrooms
            - renovated (0 or 1)
    
    Returns:
        str: Predicted price category
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # Ensure all features are present
    for feature in features:
        if feature not in input_df.columns:
            raise ValueError(f"Missing feature: {feature}")
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = rf_model.predict(input_scaled)[0]
    
    return prediction

# Example usage
example_apartment = {
    'build_area': 85,
    'construction_year': 2000,
    'floor_number': 3,
    'latitude': 37.9838,
    'longitude': 23.7275,
    'number_of_bedrooms': 2,
    'renovated': 1
}

print("\nExample prediction:")
print(f"Apartment features: {example_apartment}")
print(f"Predicted price category: {predict_price_category(example_apartment)}") 