from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import pandas as pd
import torch
import subprocess
from transformers import AutoProcessor
# Load the fine-tuned model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def inference_for_image(model, processor, image_path, device):
    # Open and convert the image
    image = Image.open(image_path).convert('RGB')

    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


# Example usage:
testing_dataset=pd.read_csv('validated_test_data_csv')

# Assuming `image` is already loaded and processed
image_path = testing_dataset.iloc[3]['image']

caption = inference_for_image(model, processor, image_path, device)
print("Generated text is: ", caption)