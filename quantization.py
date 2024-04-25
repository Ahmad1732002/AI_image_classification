import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the fine-tuned model
fine_tuned_model_path = "fine_tuned_model"
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_model")

# Define the validation dataset and DataLoader
validation_data_path = 'exp2_test_data9.csv'
validation_dataset = pd.read_csv(validation_data_path)
# Calculate model size function
def calculate_model_size(model_path):
    size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            size += os.path.getsize(os.path.join(root, file))
    return size / (1024 * 1024)  # Size in MB
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
    model, {nn.Linear}, dtype=torch.qint8)
pre_quantization_size = calculate_model_size(fine_tuned_model_path)
print(f"Model size before quantization: {pre_quantization_size:.2f} MB")
quantized_model_path="BLIPSTAT_updated"
model.save_pretrained(quantized_model_path)

post_quantization_size = calculate_model_size(quantized_model_path)
print(f"Model size after quantization: {post_quantization_size:.2f} MB")


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
test_data_path = 'exp2_test_data9.csv' 
test_dataset = pd.read_csv(test_data_path)
test_dataset = CustomDataset(test_dataset, processor)
        encoding['text'] = text

        return encoding

test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

test_accuracy = evaluate(quantized_model, test_dataloader)
print(f"Test Accuracy of Quantized Model: {test_accuracy:.2f}")
