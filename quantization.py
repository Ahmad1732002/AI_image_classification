import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm

# Load the fine-tuned model
fine_tuned_model_path = "fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)

# Define the validation dataset and DataLoader
validation_data_path = 'validation_data.csv'  # Adjust path as per your dataset
validation_dataset = pd.read_csv(validation_data_path)

# Assuming you have defined your validation dataset and DataLoader
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

# Define the quantization method and configuration
quantization_config = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8, qconfig=quantization_config)

# Evaluate the quantized model
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

# Test the quantized model
test_data_path = 'test_data.csv'  # Adjust path as per your dataset
test_dataset = pd.read_csv(test_data_path)
test_dataset = CustomDataset(test_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

test_accuracy = evaluate(quantized_model, test_dataloader)
print(f"Test Accuracy of Quantized Model: {test_accuracy:.2f}")
