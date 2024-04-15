import random
import subprocess
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration


# Install the "transformers" package
transformers_command = "pip install git+https://github.com/huggingface/transformers.git@main"
subprocess.run(transformers_command, shell=True)


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

def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  texts = [item["text"] for item in batch]
  print('this part works', texts)

  text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=20)
  print('text inputs collected')

  new_batch["labels"] = text_inputs.input_ids

  for item in batch:
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])

  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

  return new_batch


from transformers import AutoProcessor, BlipForConditionalGeneration

processor = Pix2StructProcessor.from_pretrained('google/matcha-chartqa')
model = Pix2StructForConditionalGeneration.from_pretrained('google_model2')


# train_dataset = ImageCaptioningDataset(training_dataset, processor)
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn=collator)

# validation_dataset = ImageCaptioningDataset(validation_dataset, processor)
# validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=64, collate_fn=collator)


test_dataset = ImageCaptioningDataset(testing_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64, collate_fn=collator)




import torch


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


def sample_inference(model, processor, dataset, device, num_samples=2):
    # Randomly select a few samples from the dataset
  

    # Initialize lists for images, texts, and processed inputs
    images, texts, flattened_patches_list, attention_mask_list = [], [], [], []

    for idx in len(dataset):
        # Use the __getitem__ method of your dataset to get the data
        data = dataset[idx]

        img_path = data['image_path']  # Adjust this if your dataset structure is different
        text = data['text']  # Adjust this if your dataset structure is different

        # Open and convert image
        image = Image.open(img_path).convert('RGB')
        images.append(image)
        texts.append(text)

        inputs = processor(images=image,text=' ', return_tensors="pt", max_patches=512).to(device)

        flattened_patches = inputs.flattened_patches
        attention_mask = inputs.attention_mask

        generated_ids = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


        # Process image and text separately
        # pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        # input_ids = processor(text=text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Append processed values to lists
        flattened_patches_list.append(flattened_patches)
        attention_mask_list.append(attention_mask)

    # Generate predictions for each sample
    preds, refs = [], []
    for flattened_patches, attention_mask in zip(flattened_patches_list, attention_mask_list):
        generated_ids = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        preds.extend(generated_caption)

    refs = [text.strip() for text in texts]

    return preds, refs

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
sample_preds, sample_refs = sample_inference(model, processor, test_dataset, device)
correct_matches = 0
total_samples = 0
for pred, ref in zip(sample_preds, sample_refs):
    pred_keyword = keyword_presence(pred)
    ref_keyword = keyword_presence(ref)
    if pred_keyword and ref_keyword and pred_keyword.lower() == ref_keyword.lower():
        correct_matches += 1
    total_samples += 1
accuracy = correct_matches / total_samples if total_samples > 0 else 0
print(f"Test Accuracy (based on keyword matches): {accuracy:.2f}")