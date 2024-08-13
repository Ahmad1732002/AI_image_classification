
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import clip
from transformers import CLIPProcessor, CLIPModel

# json_path = 'path to train_data.json'
# image_path = 'path to training dataset'


# with open(json_path, 'r') as f:
#     input_data = []
#     for line in f:
#         obj = json.loads(line)
#         input_data.append(obj)

training_dataset = pd.read_csv('image_labels.csv')
# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print('DEVICE USED IS: ', device)

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Define a custom dataset
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img_path = self.dataset.iloc[idx]['image']
        text = clip.tokenize(self.dataset.iloc[idx]['text'])
        image = preprocess(Image.open(img_path))

        return image, text

train_dataset = ImageCaptioningDataset(training_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)



# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
  model.float()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
num_epochs = 1
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch

        images= images.to(device)
        texts = texts.to(device)
        texts=texts.squeeze()

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

# Save the fine-tuned
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }, f"CLIP_Finetuned.pt") #just change to your preferred folder/filename
