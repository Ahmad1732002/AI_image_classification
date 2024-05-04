# An Automated Optimization Framework for Fast and Energy-Efficient Inference of Multimodal Large Language Models (MLLMs) Targeting Embedded Devices

## Overview
This project addresses the deployment challenges of Multimodal Large Language Models (MLLMs), such as the BLIP model, which are typically resource-intensive due to their substantial computational demands and memory requirements. The primary goal is to finetune and optimize MLLMs for security and privacy-related applications, focusing on distinguishing between real and AI-generated data, and providing justifying captions for classifications.

## Features
- **Model Finetuning:** Utilizes models like Salesforce's BLIP, Google's Matcha, and Microsoft's pretrained models for finetuning on a specific dataset comprising natural and AI-generated images.
- **Dynamic Quantization:** Implements dynamic quantization techniques to reduce the computational complexity and minimize memory usage, enabling deployment on resource-constrained devices like the Nvidia Jetson Xavier.
- **Demo Application:** A demo to showcase the before and after effects of model finetuning and optimization on model inference.

## Dataset
The project uses the BigGAN dataset, which includes:
- **Training Set:** 162,000 natural images and 162,000 AI-generated images.
- **Testing Set:** 6,000 natural images and 6,000 AI-generated images.
The datasets are used to train and evaluate the model performance, ensuring robustness in real-world scenarios.

## Repository Structure
- **Dataset Labeling:** Scripts for labeling the dataset are provided, generating CSV files that categorize images as natural or AI-generated.
- **Model Directories:** Each model (Salesforce/BLIP, Google/Matcha, and Microsoft) has a dedicated directory containing scripts for finetuning on the dataset and testing the accuracy.
- **Quantization:** Contains scripts for applying dynamic quantization to the finetuned models.
- **Demo:** A demo script is included to demonstrate the inference capabilities of the models before and after optimization.

## How to Run the Code
First, you need to download the GenImage dataset, which can be found [here](https://arxiv.org/abs/2306.08571). Then run the `dataset_labeling.py` script in the dataset, which will produce two CSV files: `exp2_test_data9.csv` and `exp2_train_data9.csv`. These files are necessary for running the finetuning code for one of the four models attached in the folders. The finetuning code will produce a finetuned model, which is required to run the accuracy classification as well as the inference codes provided in the above models. The finetuned model can be further optimized by applying one of the optimization techniques in the `models_optimization` folder. Additionally, there is a demo in the `inference_demo` folder that can be run to show how the performance of the finetuned model changes after optimization.
