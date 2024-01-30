import subprocess
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Install the "transformers" package
transformers_command = "pip install git+https://github.com/huggingface/transformers.git@main"
subprocess.run(transformers_command, shell=True)



training_dataset = pd.read_csv('train_data_csv')
validation_dataset = training_dataset.sample(frac=0.2)
testing_dataset=pd.read_csv('test_data_csv')

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

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"Error loading image: {img_path}")
            return None

    

        encoding = self.processor(images=image, text=text, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

train_dataset = ImageCaptioningDataset(training_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

validation_dataset = ImageCaptioningDataset(validation_dataset, processor)
validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=2)


test_dataset = ImageCaptioningDataset(testing_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=2)


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
    sample_data = [dataset[i] for i in sample_indices]

    # Process images and texts
    images = [Image.open(item['image']).convert('RGB') for item in sample_data]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)

    texts = [item['text'] for item in sample_data]
    input_ids = processor(text=texts, return_tensors="pt", padding=True, truncation=True).input_ids

    # Generate predictions
    outputs = model.generate(pixel_values=pixel_values)
    preds = processor.batch_decode(outputs, skip_special_tokens=True)
    refs = [text.strip() for text in texts]

    return preds, refs

for epoch in range(50):
    print("Epoch:", epoch)  

    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")

    for idx, batch in progress_bar:

        if batch is None:
            continue
        
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


