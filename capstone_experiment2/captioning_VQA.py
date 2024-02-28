from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import pandas as pd
import torch
import subprocess
from transformers import AutoProcessor
import os
import pandas as pd






def reasoning_caption(model, processor, image_path, device, prompt):
    # Open and convert the image
    image = Image.open(image_path).convert('RGB')
 
   

    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def general_caption_inference(model, processor, image_path, device):
    
    # Open and convert the image
    image = Image.open(image_path).convert('RGB')

    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def generate_captions(ai_image_paths, natural_image_paths):
    # Load the pretrained model and processor
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    # processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Set device (GPU if available, otherwise CPU)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    captions = []
    prompt = "Question: What details indicate this image is AI-generated? Answer:"
    #what is abnormal in the image
    for x in ai_image_paths:
        image_path = x

        general_caption= general_caption_inference(model,processor,image_path,device)
        reasoning_caption = reasoning_caption(model, processor, image_path, device, prompt)


        captions.append("This image is AI-generated, it is an image of ", general_caption, "it is AI-generated because", reasoning_caption)

    prompt = "Question: What details indicate this image is not AI-generated? Answer:"
    
    for x in natural_image_files:

        image_path = x

        caption = reasoning_caption(model, processor, image_path, device, prompt)
        captions.append("This image is natural, it is an image of ", general_caption, "it is natural because", reasoning_caption)
    return captions

   
# Path to your 'ai' folder
train_ai_folder_path = 'imagenet_ai_0419_biggan/train/ai'
train_natural_folder_path= 'imagenet_ai_0419_biggan/train/nature'

# Path to your 'ai' folder
test_ai_folder_path = 'imagenet_ai_0419_biggan/val/ai'
test_natural_folder_path= 'imagenet_ai_0419_biggan/val/nature'

train_ai_image_files = [os.path.join(train_ai_folder_path, file) for file in os.listdir(train_ai_folder_path)]
train_natural_image_files=[os.path.join(train_natural_folder_path, file) for file in os.listdir(train_natural_folder_path)]

test_ai_image_files = [os.path.join(train_ai_folder_path, file) for file in os.listdir(train_ai_folder_path)]
test_natural_image_files=[os.path.join(train_natural_folder_path, file) for file in os.listdir(train_natural_folder_path)]

def validate_images(dataset):
    valid_images = []
    for img_path in dataset:
        try:
            # Attempt to open the image and convert it to RGB to ensure it's valid
            with Image.open(img_path) as img:
                img.convert('RGB')
            valid_images.append(img_path)  # Add the path to the list of valid images
        except (IOError, FileNotFoundError) as e:
            print(f"Invalid or missing image removed: {img_path}")
    return valid_images

# Validate datasets
train_ai_image_files = validate_images(train_ai_image_files)
train_natural_image_files=validate_images(train_natural_image_files)

test_natural_image_files=validate_images(test_natural_image_files)
test_ai_image_files = validate_images(test_ai_image_files)



train_captions = generate_captions(train_ai_image_files,train_natural_image_files)
test_captions= generate_captions(test_ai_image_files,test_natural_image_files)

train_image_files= train_ai_image_files + train_natural_image_files
test_image_files= test_ai_image_files + test_natural_image_files


data_train = {'image': train_image_files, 'text': train_captions}
df_train = pd.DataFrame(data_train)


data_test = {'image': test_image_files, 'text': test_captions}
df_test = pd.DataFrame(data_test)


# Save to a CSV file if needed
df_train.to_csv('./exp2_train_data', index=False)
df_test.to_csv('./exp2_test_data', index=False)