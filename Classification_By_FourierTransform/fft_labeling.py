from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import pandas as pd
import torch
import subprocess
from transformers import AutoProcessor
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_pattern_name(file_names):
    result = []
    for file_name in file_names:
        # Extract the part before 'image'
        base_name = os.path.basename(file_name)
        parts = base_name.split('_')
        if 'image' in parts:
            index = parts.index('image')
            words_before_image = ' '.join(parts[1:index])
            result.append(words_before_image)
    return result


def reasoning_caption(model, tokenizer, images, device,prompt):
    batch_size = 64
    all_generated_text = []

    # Preparing batches of images
    n_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)

    for batch_index in range(n_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(images))
        current_batch_images = images[start_index:end_index]

        batch_captions = []
        for image in current_batch_images:
            # Assuming the model has a method like 'answer_question' to generate the caption
            encoded_answer = model.answer_question(
                image,
                prompt,
                tokenizer
            )
            # Decode the answer
            decoded_answer = tokenizer.decode(encoded_answer, skip_special_tokens=True)
            batch_captions.append(decoded_answer)

        all_generated_text.extend(batch_captions)  # Store captions from current batch

    return all_generated_text


def general_caption_inference(model, processor, images, device):

    batch_size = 64
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
    # # Load the pretrained model and processor

    model_id = "qresearch/llama-3-vision-alpha-hf"
    model2 = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )

    # Set device (GPU if available, otherwise CPU)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model2.to(device)

    captions = []
    prompt = "This is the fourier transform of an AI generated image. What patterns in this fourier transform indicate that the image is AI-generated. Give a concise one sentence answer."

    images = [Image.open(path).convert("RGB") for path in ai_image_paths]
    injected_patterns = extract_pattern_name(ai_image_paths)
    # general_caption= general_caption_inference(model,processor,images,device)
    reasoning= reasoning_caption(model2, tokenizer, images, device,prompt)
    for x in range(len(reasoning)):
        # general_caption[x]=general_caption[x].strip()
        reasoning[x]=reasoning[x].strip()
        captions.append(f"The fourier transform of the image indicate that it is AI-generated. {reasoning[x]} There is also the {injected_patterns[x]} pattern in the fourier transform which is a common artifact in AI-generated images")






    prompt2 = "This is the fourier transform of a real image. What patterns in this image indicate that the image is real. Give a concise one sentence answer."

    images = [Image.open(path).convert("RGB") for path in natural_image_paths]
    # general_caption= general_caption_inference(model,processor,images,device)
    reasoning= reasoning_caption(model2, tokenizer, images, device, prompt2)
    for x in range(len(reasoning)):
        # general_caption[x]=general_caption[x].strip()
        reasoning[x]=reasoning[x].strip()
        captions.append(f"the fourier transform of the image indicate that it is real. {reasoning[x]}")
    return captions


# Path to your 'ai' folder
train_ai_folder_path = 'CIFAKE_fake_fft'
train_natural_folder_path= 'CIFAKE_real_fft'



train_ai_image_files = [os.path.join(train_ai_folder_path, file) for file in os.listdir(train_ai_folder_path)]
train_natural_image_files=[os.path.join(train_natural_folder_path, file) for file in os.listdir(train_natural_folder_path)]




train_captions = generate_captions(train_ai_image_files,train_natural_image_files)

train_image_files= train_ai_image_files + train_natural_image_files



data_train = {'image': train_image_files, 'text': train_captions}
df_train = pd.DataFrame(data_train)





# # Save to a CSV file if needed
df_train.to_csv('./fft_explained.csv', index=False)
