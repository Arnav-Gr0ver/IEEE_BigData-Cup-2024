import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import MultimodalIntentionModel
from utils import load_data, preprocess_text, preprocess_image
from torch.optim import Adam

# Load dataset
train_data = load_data('data/train.json')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create DataLoader for batching
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Initialize model
model = MultimodalIntentionModel(num_intentions=10)
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        text = batch['text']
        images = batch['image']
        labels = batch['intention_labels']

        # Preprocess text and images
        text_input = preprocess_text(text, tokenizer)
        image_input = preprocess_image(images)

        # Forward pass
        output = model(text_input, image_input)

        # Compute loss
        loss = criterion(output, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
