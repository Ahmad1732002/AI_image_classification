import random
import subprocess
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Install the "transformers" package
transformers_command = "pip install git+https://github.com/huggingface/transformers.git@main"
subprocess.run(transformers_command, shell=True)

#train_dataset = pd.read_csv('validated_train_data_csv')
#training_dataset= train_dataset.sample(frac=0.1)
#validation_dataset = train_dataset.sample(frac=0.05)

training_dataset = pd.read_csv('exp2_train_data9.csv')
validation_dataset = training_dataset.sample(frac=0.2)
testing_dataset=pd.read_csv('exp2_test_data9.csv')

from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
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
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding['image_path'] = img_path  # Add image path to the encoding
        encoding['text'] = text  # Add text to the encoding
        
        return encoding

from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

train_dataset = ImageCaptioningDataset(training_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

validation_dataset = ImageCaptioningDataset(validation_dataset, processor)
validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=64)


test_dataset = ImageCaptioningDataset(testing_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64)


def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model.generate(pixel_values=pixel_values)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)
            refs = processor.batch_decode(input_ids, skip_special_tokens=True)

            for pred, ref in zip(preds, refs):
                if pred.strip() == ref.strip():
                    correct += 1
                total += 1
    return correct / total


import torch



optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()


def sample_inference(model, processor, dataset, device, num_samples=2):
    # Randomly select a few samples from the dataset
    sample_indices = random.sample(range(len(dataset)), num_samples)

    # Initialize lists for images, texts, and processed inputs
    images, texts, pixel_values_list, input_ids_list = [], [], [], []

    for idx in sample_indices:
        # Use the __getitem__ method of your dataset to get the data
        data = dataset[idx]
      
        img_path = data['image_path']  # Adjust this if your dataset structure is different
        text = data['text']  # Adjust this if your dataset structure is different

        # Open and convert image
        image = Image.open(img_path).convert('RGB')
        images.append(image)
        texts.append(text)

        # Process image and text separately
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        input_ids = processor(text=text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Append processed values to lists
        pixel_values_list.append(pixel_values)
        input_ids_list.append(input_ids)

    # Generate predictions for each sample
    preds, refs = [], []
    for pixel_values, input_ids in zip(pixel_values_list, input_ids_list):
        outputs = model.generate(pixel_values=pixel_values, input_ids=input_ids)
        pred = processor.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(pred)

    refs = [text.strip() for text in texts]

    return preds, refs



for epoch in range(1):
    print("Epoch:", epoch)  

    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")

    for idx, batch in progress_bar:
        
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
        
        loss = outputs.loss

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

       
     # Sample inference at the end of each epoch
    sample_preds, sample_refs = sample_inference(model, processor, train_dataset, device)
    for pred, ref in zip(sample_preds, sample_refs):
        print(f"Sample Prediction: {pred}\nReference: {ref}")



    train_accuracy = calculate_accuracy(model, validation_dataloader, device)
    print(f"Training Accuracy after epoch {epoch}: {train_accuracy}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")

# Save optimizer's state_dict
torch.save(optimizer.state_dict(), "optimizer_state.pth")

#load test data and calculate accuracy 
def test_model_and_calculate_accuracy(model, dataloader, processor, device):
    model.eval()
    correct_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            # Generate captions
            outputs = model.generate(pixel_values=pixel_values)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)

            # Get reference captions
            refs = processor.batch_decode(input_ids, skip_special_tokens=True)

            # Compare generated captions with reference captions
            for pred, ref in zip(preds, refs):
                if pred.strip().lower() == ref.strip().lower():
                    correct_matches += 1
                total_samples += 1

    accuracy = correct_matches / total_samples if total_samples > 0 else 0
    return accuracy

# Test the model and calculate accuracy
test_accuracy = test_model_and_calculate_accuracy(model, test_dataloader, processor, device)
print(f"Test Accuracy (based on exact matches): {test_accuracy:.2f}")
