from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import pandas as pd
import torch
import subprocess
from transformers import AutoProcessor
import os
import pandas as pd





def reasoning_caption(model, processor, images, device, prompt):
    batch_size = 256
    all_generated_text = []

    # Preparing batches of images
    n_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)

    for batch_index in range(n_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(images))
        current_batch_images = images[start_index:end_index]

        # Processing a batch of images
        #inputs = processor(current_batch_images, text=[prompt] * len(current_batch_images), return_tensors="pt", padding=True).to(device, torch.float16)
        inputs = processor(current_batch_images, text=[prompt] * len(current_batch_images), return_tensors="pt", padding=True).to(device, torch.float16)

        # Generate text for the batch
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_generated_text.extend(generated_text)

    return all_generated_text


def general_caption_inference(model, processor, images, device):

    batch_size = 256
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
        inputs = processor(current_batch_images, return_tensors="pt").to(device, torch.float16)
        
        # Generate text
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_generated_text.extend(generated_text)
        
    return all_generated_text

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
    # for x in ai_image_paths:
    #     image_path = x
        
    #      # Open and convert the image
    #     image = Image.open(image_path).convert('RGB')

    #     general_caption= general_caption_inference(model,processor,image,device)
    #     reasoning= reasoning_caption(model, processor, image_path, device, prompt)


    #     captions.append(f"This image is AI-generated, it is an image of {general_caption}, it is AI-generated because {reasoning}")

   
   
    images = [Image.open(path).convert("RGB") for path in ai_image_paths]
    general_caption= general_caption_inference(model,processor,images,device)
    reasoning= reasoning_caption(model, processor, images, device, prompt)
    for x in range(len(general_caption)):
        general_caption[x]=general_caption[x].strip()
        reasoning[x]=reasoning[x].strip()
        captions.append(f"This image is AI-generated, it is an image of {general_caption[x]}, it is AI-generated because {reasoning[x]}")
    
    

  
  


    prompt = "Question: What details indicate this image is not AI-generated? Answer:"
    
    images = [Image.open(path).convert("RGB") for path in ai_image_paths]
    general_caption= general_caption_inference(model,processor,images,device)
    reasoning= reasoning_caption(model, processor, images, device, prompt)
    for x in range(len(general_caption)):
        general_caption[x]=general_caption[x].strip()
        reasoning[x]=reasoning[x].strip()
        captions.append(f"This image is natural, it is an image of {general_caption[x]}, it is natural because {reasoning[x]}")
    return captions

   
# Path to your 'ai' folder
train_ai_folder_path = 'imagenet_ai_0419_biggan/train/ai'
train_natural_folder_path= 'imagenet_ai_0419_biggan/train/nature'

# Path to your 'ai' folder
test_ai_folder_path = 'imagenet_ai_0419_biggan/val/ai'
test_natural_folder_path= 'imagenet_ai_0419_biggan/val/nature'

train_ai_image_files = [os.path.join(train_ai_folder_path, file) for file in os.listdir(train_ai_folder_path)]
train_natural_image_files=[os.path.join(train_natural_folder_path, file) for file in os.listdir(train_natural_folder_path)]

test_ai_image_files = [os.path.join(test_ai_folder_path, file) for file in os.listdir(test_ai_folder_path)]
test_natural_image_files=[os.path.join(test_natural_folder_path, file) for file in os.listdir(test_natural_folder_path)]

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
print("the train captions length is: ", train_captions)

test_captions= generate_captions(test_ai_image_files,test_natural_image_files)
print("the test captions length is: ", test_captions)
train_image_files= train_ai_image_files + train_natural_image_files
test_image_files= test_ai_image_files + test_natural_image_files

print("train image files size is: ", train_image_files)
print("test image files size is: ", test_image_files)


data_train = {'image': train_image_files, 'text': train_captions}
df_train = pd.DataFrame(data_train)


data_test = {'image': test_image_files, 'text': test_captions}
df_test = pd.DataFrame(data_test)


# Save to a CSV file if needed
df_train.to_csv('./exp2_train_data5.csv', index=False)
df_test.to_csv('./exp2_test_data5.csv', index=False)
