import random
import subprocess
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration


# Assuming the "transformers" package is already installed and datasets are loaded

# Define your ImageCaptioningDataset, train, validation, and test dataloaders as before

testing_dataset=pd.read_csv('exp2_test_data9.csv')

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("BLIP_finetuned")
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

# Assuming train_dataset, validation_dataset, and test_dataset setup as before
test_dataset = ImageCaptioningDataset(testing_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)

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
    model.eval()
    correct_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            # Generate captions
            outputs = model.generate(pixel_values=pixel_values, max_length=120)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)

            # Get reference captions
            refs = [batch['text'][i] for i in range(len(batch['text']))]  # Assuming 'text' key exists and holds the reference captions

            # Compare generated captions with reference captions
            for pred, ref in zip(preds, refs):
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
