from transformers import BlipForConditionalGeneration
import pandas as pd
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from torchvision.transforms import functional as F

model = BlipForConditionalGeneration.from_pretrained("fine_tuned_model")
# Specify the device
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

testing_dataset=pd.read_csv('validated_test_data_csv')

# Assuming `image` is already loaded and processed
img_path = testing_dataset.iloc[3]['image']
image = Image.open(img_path).convert('RGB')
image = F.resize(image, [224, 224])  # Resize image to the required input size

prompt = "Question: Is this image natural or AI generated and why? Answer:"

inputs = processor(image, text=prompt, return_tensors="pt", padding=True, truncation=True).to(device)

# Generate text
generated_ids = model.generate(**inputs, max_length=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
#
