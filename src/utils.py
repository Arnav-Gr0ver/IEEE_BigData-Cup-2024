import torch
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import json

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_text(text, tokenizer):
    # Tokenize text and convert to tensor
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    return encoding

def preprocess_image(image_path):
    # Preprocess image for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model
