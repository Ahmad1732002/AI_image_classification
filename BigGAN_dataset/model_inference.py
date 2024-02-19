from transformers import BlipForConditionalGeneration
import pandas as pd
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
# Load the fine-tuned model
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_model")
# Specify the device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

testing_dataset=pd.read_csv('validated_test_data_csv')


def question_answering(model, image_path, question, device):
    model.eval()
    with torch.no_grad():
        # Perform inference
        image = Image.open(image_path).convert('RGB')
        pixel_valuess = processor(images=image, return_tensors="pt").pixel_values.to(device)

        outputs = model(pixel_values=pixel_valuess, inputs=question, return_tensors="pt")
        answer_ids = outputs.logits.argmax(-1)
        answer = processor.decode(answer_ids.squeeze(), skip_special_tokens=True)
    return answer

# Assuming `image` is already loaded and processed
img_path = testing_dataset.iloc[3]['image']

question = 'Is this image natural or AI generated and why?'
answer = question_answering(model, img_path, question, device)
print('Answer:', answer)


# # Specify the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Load optimizer's state_dict
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# optimizer.load_state_dict(torch.load("optimizer_state.pth"))
# testing_dataset=pd.read_csv('validated_test_data_csv')

# # Provide an image to the model for captioning
# def generate_caption(model, processor, image_path, device):
#     image = Image.open(image_path).convert('RGB')
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

#     # Generate caption
#     outputs = model.generate(pixel_values=pixel_values)
#     caption = processor.batch_decode(outputs, skip_special_tokens=True)

#     return caption

# # Example usage:
# img_path = testing_dataset.iloc[3]['image']
# caption = generate_caption(model, processor, img_path, device)
# print("Generated Caption:", caption[0])