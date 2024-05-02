import random
import subprocess
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
# Assuming the "transformers" package is already installed and datasets are loaded

# Define your ImageCaptioningDataset, train, validation, and test dataloaders as before

testing_dataset=pd.read_csv('exp2_test_data9.csv')

processor = Pix2StructProcessor.from_pretrained('google/matcha-chartqa')
model = Pix2StructForConditionalGeneration.from_pretrained('google/matcha-chartqa')


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
    
    
def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  texts = [item["text"] for item in batch]
#   print('this part works', texts)

  text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=20)
  print('helloooooo: text inputs collected')

  new_batch["labels"] = text_inputs.input_ids

  for item in batch:
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])

  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

  return new_batch

test_dataset = ImageCaptioningDataset(testing_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8, collate_fn=collator)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)



def keyword_presence(text, keywords=['natural', 'AI'], match_length=30):
    """
    Check if the text contains any of the specified keywords within the first `match_length` characters, ignoring case sensitivity.
    Returns the keyword found or None if no keyword is found.
    """
    text_lower = text[:match_length].lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return keyword
    return None

def test_model_and_calculate_accuracy(model, dataloader, processor, device):
    # Set the model to evaluation mode
    # model.eval()
    correct_matches = 0
    total_samples = 0
    
    # Initialize lists to store predictions and references
    all_preds, all_refs = [], []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for batch in dataloader:
            # Assuming the dataloader provides 'image_paths' and 'text' in each batch
            image_paths = batch['image_path']  # Modify as per your dataset structure
            texts = batch['text']  # Modify as per your dataset structure
            
            # Load and process images in batch
            images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
            inputs = processor(images=images, return_tensors="pt", max_patches=512).to(device)
            
            flattened_patches = inputs['flattened_patches']
            attention_mask = inputs['attention_mask']
            
            # Generate captions for the batch
            generated_ids = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=50)
            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Get reference captions
            refs = [batch['text'][i] for i in range(len(batch['text']))]  # Assuming 'text' key exists and holds the reference captions

            # Compare generated captions with reference captions
            for pred, ref in zip(generated_captions, refs):
                pred_keyword = keyword_presence(pred)
                ref_keyword = keyword_presence(ref)
                if pred_keyword and ref_keyword and pred_keyword.lower() == ref_keyword.lower():
                    correct_matches += 1
                total_samples += 1

    accuracy = correct_matches / total_samples if total_samples > 0 else 0
    return accuracy
# Adapt other parts of your code as necessary for your model training and evaluation loop

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example test of the adapted accuracy calculation
test_accuracy = test_model_and_calculate_accuracy(model, test_dataloader, processor, device)
print(f"Test Accuracy (based on keyword matches): {test_accuracy:.2f}")
