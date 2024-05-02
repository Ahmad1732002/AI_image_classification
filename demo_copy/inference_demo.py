from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import pandas as pd
import torch
import subprocess
from PIL import Image
import time
from transformers import BlipProcessor, BlipForConditionalGeneration

import os

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_model")

# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def general_caption_inference(model, processor, images, device):

    batch_size = 32
    n_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)
    all_generated_text=[]

# Iterate through each batch
    for batch_index in range(n_batches):
        # Calculate start and end indices of the current batch
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(images))
        
        # Select the current batch of images
        current_batch_images = images[start_index:end_index]
        
        # Assuming `processor` is your image processor and `model` is your text generation model
        # Process images
        pixel_values = processor(current_batch_images, return_tensors="pt").to(device, torch.float16)
        
        # Generate text
        outputs = model.generate(**pixel_values, max_length=128)
        caption = processor.batch_decode(outputs, skip_special_tokens=True)
        all_generated_text.extend(caption)
        
    return all_generated_text



# Path to the folder containing the images
folder_path = './demo/AI_images'

# List all files in the directory
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]  # Adjust the tuple to match your image file types

# Load images and convert them to RGB
images = [Image.open(os.path.join(folder_path, file)).convert("RGB") for file in image_files]

# Now, `images` is a list of PIL Image objects




# Measure time taken for caption generation
start_time = time.time()
captions = general_caption_inference(model, processor, images, device)
end_time = time.time()



# Calculate and print execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")
# natural_img_paths=[]
# for i in range(6999,7005):
#     natural_img_paths.append(testing_dataset.iloc[i]['image'])

# nat_images = [Image.open(path).convert("RGB") for path in natural_img_paths]

# nat_captions = general_caption_inference(model, processor, nat_images, device)

# Since `captions` is a list of captions for each image, you might want to print them differently.
# For example, print the first few captions to check:
print('BLIP MODEL')
for i, caption in enumerate(captions[:5]):
    print(f"Image {i+1} Caption: {caption}")

# for i, caption in enumerate(nat_captions[:5]):
#     print(f"Image {i+5} Caption: {caption}")
