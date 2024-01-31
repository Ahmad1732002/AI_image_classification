import os
import pandas as pd

# Path to your 'ai' folder
train_ai_folder_path = 'imagenet_ai_0419_biggan/train/ai'
train_natural_folder_path= 'imagenet_ai_0419_biggan/train/nature'

# Step 1: List all images in the train ai and nature folder
ai_image_files = [os.path.join(train_ai_folder_path, file) for file in os.listdir(train_ai_folder_path)]
natural_image_files=[os.path.join(train_natural_folder_path, file) for file in os.listdir(train_natural_folder_path)]




# Step 2: Write or load your captions here
# Assuming you have a list of captions in the same order as the images
captions = []
for i in range(len(ai_image_files)):
    captions.append('This image is AI generated')

for j in range(len(natural_image_files)):
    captions.append('This image is natural image')

# Make sure the number of captions matches the number of images
assert len(ai_image_files) + len(natural_image_files) == len(captions), "Number of images and captions should be the same"

image_files= ai_image_files + natural_image_files

# Step 3: Create a pandas DataFrame
data = {'image': image_files, 'text': captions}
df = pd.DataFrame(data)

# Display the DataFrame to verify it's correct
print(df.head())

# Save to a CSV file if needed
df.to_csv('./train_data_csv', index=False)

