import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from time import time
print("GOOGLE ORIGINAL INFERENCE")
# Load dataset (assuming a column 'image' exists)
testing_dataset = pd.read_csv('exp2_test_data9.csv')[:1000]  # Using only the first 1000 images for inference
nat_dataset=pd.read_csv('exp2_test_data9.csv')[6999:7005]
# Define the custom dataset class
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image']
        image = Image.open(img_path).convert('RGB')
        return image

# Custom collate function
def collate_fn(batch):
    # Process a batch of images
    images = [item for item in batch]
    return images

# Initialize the processor and model
processor = Pix2StructProcessor.from_pretrained('google/matcha-chartqa')
model = Pix2StructForConditionalGeneration.from_pretrained('google_model')

# Create a dataset and data loader
test_dataset = ImageCaptioningDataset(testing_dataset, processor)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

natural_dataset = ImageCaptioningDataset(nat_dataset, processor)
natural_dataloader = DataLoader(natural_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to perform inference and measure time
def perform_inference(dataloader, model, processor, device):
    model.eval()
    start_time = time()
    captions = []
    with torch.no_grad():
        for images in dataloader:
            inputs = processor(images=images,text=' ', return_tensors="pt", max_patches=512).to(device)
            outputs = model.generate(**inputs, max_length=50)
            caption = processor.batch_decode(outputs,text=' ', skip_special_tokens=True)
            captions.append(caption[0])
    end_time = time()
    return captions, end_time - start_time

# Run inference
captions, inference_time = perform_inference(test_dataloader, model, processor, device)
nat_captions, nat_inference_time = perform_inference(test_dataloader, model, processor, device)
print("GOOGLE MODEL")
print(f"Total Inference Time for 1000 Images: {inference_time:.2f} seconds")
#print("First 5 captions:", captions[:5])
#print("Last 5 captions:", captions[-5:])
for i, caption in enumerate(captions[:5]):
    print(f"Image {i+1} Caption: {caption}")

for i, caption in enumerate(nat_captions[:5]):
    print(f"Image {i+6} Caption: {caption}")