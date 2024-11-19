import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models
import torchvision.transforms as T
from PIL import Image

class MultimodalIntentionModel(nn.Module):
    def __init__(self, num_intentions=10):
        super(MultimodalIntentionModel, self).__init__()
        
        # Text model (BERT)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_fc = nn.Linear(self.bert_model.config.hidden_size, 512)
        
        # Image model (ResNet)
        self.resnet_model = models.resnet18(pretrained=True)
        self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, 512)
        
        # Fusion layer
        self.fc = nn.Linear(512 + 512, num_intentions)

    def forward(self, text_input, image_input):
        # Process text with BERT
        text_output = self.bert_model(**text_input).pooler_output
        text_output = self.bert_fc(text_output)
        
        # Process image with ResNet
        image_output = self.resnet_model(image_input)
        
        # Combine text and image features
        combined_features = torch.cat((text_output, image_output), dim=1)
        
        # Final prediction
        output = self.fc(combined_features)
        
        return output
