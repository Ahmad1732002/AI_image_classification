import random
import subprocess
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration


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
model = Pix2StructForConditionalGeneration.from_pretrained('google/matcha-chartqa')


train_dataset = ImageCaptioningDataset(training_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=collator)

validation_dataset = ImageCaptioningDataset(validation_dataset, processor)
validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=8, collate_fn=collator)


test_dataset = ImageCaptioningDataset(testing_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8, collate_fn=collator)


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
# Initialize both processors
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
pix2struct_processor = Pix2StructProcessor.from_pretrained('google/matcha-chartqa')

# Load an image
image_path = testing_dataset.iloc[3]['image']
image = Image.open(image_path).convert('RGB')

# Process the image with BLIP processor
blip_output = blip_processor(images=image, return_tensors="pt")
print("BLIP Processor Output Keys:", blip_output.keys())

# Assuming Pix2StructProcessor has a similar interface
# Note: This is a hypothetical example, adjust based on actual processor methods
pix2struct_output = pix2struct_processor(images=image,text=" ", return_tensors="pt")
print("Pix2Struct Processor Output Keys:", pix2struct_output.keys())
model.train()


def sample_inference(model, processor, dataset, device, num_samples=2):
    # Randomly select a few samples from the dataset
    sample_indices = random.sample(range(len(dataset)), num_samples)

    # Initialize lists for images, texts, and processed inputs
    images, texts, flattened_patches_list, attention_mask_list = [], [], [], []

    for idx in sample_indices:
        # Use the __getitem__ method of your dataset to get the data
        data = dataset[idx]

        img_path = data['image_path']  # Adjust this if your dataset structure is different
        text = data['text']  # Adjust this if your dataset structure is different

        # Open and convert image
        image = Image.open(img_path).convert('RGB')
        images.append(image)
        texts.append(text)

        inputs = processor(images=image, return_tensors="pt", max_patches=512).to(device)

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



for epoch in range(5):
    print("Epoch:", epoch)

    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")

    for idx, batch in progress_bar:

        
        
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())


     # Sample inference at the end of each epoch
    # Save the fine-tuned model
    model.save_pretrained("google_model")

    # Save optimizer's state_dict
    torch.save(optimizer.state_dict(), "optimizer_state.pth")
    sample_preds, sample_refs = sample_inference(model, processor, train_dataset, device)
    for pred, ref in zip(sample_preds, sample_refs):
        print(f"Sample Prediction: {pred}\nReference: {ref}")



    # train_accuracy = calculate_accuracy(model, validation_dataloader, device)
    # print(f"Training Accuracy after epoch {epoch}: {train_accuracy}")

# Save the fine-tuned model
model.save_pretrained("google_model3")

# Save optimizer's state_dict
torch.save(optimizer.state_dict(), "optimizer_state.pth")

