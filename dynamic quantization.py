import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForSequenceClassification

# Load the fine-tuned model
fine_tuned_model_path = "fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)

# Measure model size before quantization
def model_size(model):
    torch.save(model.state_dict(), 'temp_model.pt')
    size = os.path.getsize('temp_model.pt')
    os.remove('temp_model.pt')
    return size / (1024 ** 2)  # size in MB

original_size = model_size(model)
print(f"Original Model Size: {original_size:.2f} MB")

# Define the validation dataset and DataLoader
validation_data_path = 'validation_data.csv'
validation_dataset = pd.read_csv(validation_data_path)

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

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8)

# Measure model size after quantization
quantized_size = model_size(quantized_model)
print(f"Quantized Model Size: {quantized_size:.2f} MB")

# Evaluate the quantized model
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = labels.to(model.device)
            outputs = model(**inputs).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Test the quantized model
test_data_path = 'exp2_test_data9.csv'
test_dataset = pd.read_csv(test_data_path)
test_dataset = CustomDataset(test_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

test_accuracy = evaluate(quantized_model, test_dataloader)
print(f"Test Accuracy of Quantized Model: {test_accuracy:.2f}")
