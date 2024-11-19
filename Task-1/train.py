import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from preprocess import calculate_features, load_graph

# Load the training data and create features
train_graph = load_graph('data/train.txt')
train_pairs = calculate_features('data/train.txt', train_graph)
train_pairs['label'] = 1  # Set label for training (you should ideally label this correctly from your data)

# Split into features (X) and labels (y)
X = train_pairs.drop(columns=['source', 'target', 'label'])
y = train_pairs['label']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
from sklearn.metrics import f1_score
print("F1 Score:", f1_score(y_val, y_pred))

# Save the model for future use
joblib.dump(model, 'models/retweet_model.pkl')
