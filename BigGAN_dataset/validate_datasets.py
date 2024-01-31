import pandas as pd
from PIL import Image

def validate_images(dataset):
    valid_dataset = dataset.copy()
    for index, row in dataset.iterrows():
        img_path = row['image']
        try:
            Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            valid_dataset = valid_dataset.drop(index)
            print(f"Invalid image removed: {img_path}")
    return valid_dataset

# Read your datasets
training_dataset = pd.read_csv('train_data_csv')
testing_dataset = pd.read_csv('test_data_csv')

# Validate datasets
training_dataset = validate_images(training_dataset)
testing_dataset = validate_images(testing_dataset)

# Save the validated datasets to new CSV files
training_dataset.to_csv('./validated_train_data_csv', index=False)
testing_dataset.to_csv('./validated_test_data_csv', index=False)
