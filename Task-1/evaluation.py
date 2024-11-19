import pandas as pd
import joblib
from preprocess import calculate_features, load_graph

# Function to evaluate and save results
def evaluate():
    # Load the test data
    test_graph = load_graph('data/train.txt')  # Using train.txt for graph, not for labels
    test_pairs = calculate_features('data/test.txt', test_graph)
    
    # Load the trained model
    model = joblib.load('models/retweet_model.pkl')
    
    # Predict on the test data
    test_pairs['prediction'] = model.predict(test_pairs.drop(columns=['source', 'target']))
    
    # Save the results as result.json
    test_pairs['prediction'].to_json('results/result.json', orient='values')

if __name__ == "__main__":
    evaluate()
