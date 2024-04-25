import os
import torch
import torch.nn.utils.prune as prune
from transformers import BlipForConditionalGeneration

# Load your pre-trained/fine-tuned model
model_path = "fine_tuned_model"
model = BlipForConditionalGeneration.from_pretrained(model_path)

# Specify layers to prune: This example prunes the first attention layer weights in the encoder and decoder
# You might want to extend this to more layers or adjust based on model specifics and your needs
layers_to_prune = [
    (model.text_model.encoder.layers[0].self_attn.self_attn.project, 'weight'),
    (model.vision_model.encoder.layers[0].self_attn.self_attn.project, 'weight'),
]

# Applying pruning to specified layers
for layer, param in layers_to_prune:
    prune.l1_unstructured(layer, name=param, amount=0.2)  # Prune 20% of the weights

# Function to calculate the size of the model
def calculate_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    model_size = os.path.getsize("temp_model.pth") / (1024 * 1024)  # Size in MB
    os.remove("temp_model.pth")
    return model_size

pre_pruning_size = calculate_model_size(model)
print(f"Model size before pruning: {pre_pruning_size:.2f} MB")

# Optionally make pruning permanent and re-measure model size
for layer, param in layers_to_prune:
    prune.remove(layer, param)  # Make pruning permanent

post_pruning_size = calculate_model_size(model)
print(f"Model size after pruning: {post_pruning_size:.2f} MB")

# Save the pruned model
pruned_model_path = "pruned_model"
model.save_pretrained(pruned_model_path)