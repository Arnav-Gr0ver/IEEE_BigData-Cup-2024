import torch
import json
from utils import load_data, load_model, preprocess_text, preprocess_image
from model import MultimodalIntentionModel

# Load test data and model
test_data = load_data('data/test.json')
model = load_model('model/model.pth')

# Generate predictions
results = []
for instance in test_data:
    text = instance['text']
    image = instance['image']

    # Preprocess text and image
    text_input = preprocess_text(text)
    image_input = preprocess_image(image)

    # Forward pass
    with torch.no_grad():
        output = model(text_input, image_input)
    
    # Convert output to binary predictions based on BERT scores
    predictions = torch.sigmoid(output) > 0.5
    predictions = predictions.cpu().numpy()

    # Store results for each instance
    result_instance = {}
    for i, pred in enumerate(predictions[0]):
        result_instance[f"Intention {i+1}"] = "" if pred == 0 else f"Intention {i+1} Detected"
    
    results.append(result_instance)

# Save the result in result.json
with open('result.json', 'w') as f:
    json.dump(results, f, indent=4)
