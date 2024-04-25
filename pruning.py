<<<<<<< HEAD
import os
import torch
import torch.nn.utils.prune as prune
from transformers import BlipForConditionalGeneration

# Load your pre-trained/fine-tuned model
model_path = "fine_tuned_model"
model = BlipForConditionalGeneration.from_pretrained(model_path)

# Specify layers to prune: This example prunes the first attention layer weights in the encoder and decoder
# You might want to extend this to more layers or adjust based on model specifics and your needs
layers_to_prune = [
    (model.text_model.encoder.layers[0].self_attn.self_attn.project, 'weight'),
    (model.vision_model.encoder.layers[0].self_attn.self_attn.project, 'weight'),
]

# Applying pruning to specified layers
for layer, param in layers_to_prune:
    prune.l1_unstructured(layer, name=param, amount=0.2)  # Prune 20% of the weights

# Function to calculate the size of the model
def calculate_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    model_size = os.path.getsize("temp_model.pth") / (1024 * 1024)  # Size in MB
    os.remove("temp_model.pth")
    return model_size

pre_pruning_size = calculate_model_size(model)
print(f"Model size before pruning: {pre_pruning_size:.2f} MB")

# Optionally make pruning permanent and re-measure model size
for layer, param in layers_to_prune:
    prune.remove(layer, param)  # Make pruning permanent

post_pruning_size = calculate_model_size(model)
print(f"Model size after pruning: {post_pruning_size:.2f} MB")

# Save the pruned model
pruned_model_path = "pruned_model"
model.save_pretrained(pruned_model_path)
=======
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm
from torch.nn.utils import prune

# Load the fine-tuned model
fine_tuned_model_path = "fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)

class CustomDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset.iloc[idx]['image']
        text = self.dataset.iloc[idx]['text']

        image = Image.open(img_path).convert('RGB')

        encoding = self.processor(images=image, text=text, padding="max_length", return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding['image_path'] = img_path
        encoding['text'] = text

        return encoding

# Initialize the processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
validation_dataset = CustomDataset(validation_dataset, processor)
validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=32)

# Define the pruning method (magnitude-based)
def apply_magnitude_pruning(model, pruning_percentage):
    parameters_to_prune = [(module, "weight") for _, module in model.named_modules() if isinstance(module, nn.Linear)]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_percentage,
    )

# Apply pruning
pruning_percentage = 0.2  # Example: prune 20% of weights
apply_magnitude_pruning(model, pruning_percentage)

# Evaluate the pruned model
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Test the pruned model
test_data_path = 'exp2_test_data9.csv' 
test_dataset = pd.read_csv(test_data_path)
test_dataset = CustomDataset(test_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

test_accuracy = evaluate(model, test_dataloader)
print(f"Test Accuracy of Pruned Model: {test_accuracy:.2f}")
>>>>>>> a95541ccd3be65cfc4a59586e14ce3e95a693467
