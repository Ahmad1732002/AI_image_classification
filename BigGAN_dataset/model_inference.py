from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import pandas as pd
import torch
import subprocess
from transformers import AutoProcessor
# Load the fine-tuned model and processor
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_model")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def inference_for_image(model, processor, image_path, device):
    # Open and convert the image
    image = Image.open(image_path).convert('RGB')

    # Process the image
    #pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    prompt = "XYZ "

    inputs = processor(image, prompt, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_length=100)
    generated_text = processor.decode(out[0], special_tokens=True)
    return generated_text

    # # Generate caption
    # outputs = model.generate(pixel_values=pixel_values)
    # caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]  # Assuming one image is processed at a time

    # return caption

# Example usage:
testing_dataset=pd.read_csv('validated_test_data_csv')

# Assuming `image` is already loaded and processed
image_path = testing_dataset.iloc[3]['image']

caption = inference_for_image(model, processor, image_path, device)
print("Gen", caption)