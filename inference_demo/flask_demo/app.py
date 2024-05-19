from flask import Flask, request, jsonify, render_template
import os
import base64
from PIL import Image
import io
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import torch.nn as nn
from threading import Thread
from queue import Queue
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.logger.setLevel(logging.INFO)

# Load the model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("fine_tuned_model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Quantize the model dynamically
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8)
quantized_model.to(device)

# Function to calculate model size by serializing to a buffer
def calculate_model_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size = buffer.tell()  # Get the size of the buffer
    size_in_mb = size / (1024 * 1024)  # Convert bytes to MB
    formatted_size_in_mb = "{:.2f}".format(size_in_mb)  # Format to two decimal places
    return formatted_size_in_mb

model_size = calculate_model_size(model)
quantized_model_size = calculate_model_size(quantized_model)

# Request processing queue
request_queue = Queue()
result_queue = Queue()

def worker():
    while True:
        model_to_use, image_data = request_queue.get()
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pixel_values = processor(images=[image], return_tensors="pt").to(device)
            outputs = model_to_use.generate(**pixel_values, max_length=128)
            captions = processor.batch_decode(outputs, skip_special_tokens=True)
            result = captions[0] if captions else "No caption generated"
        except Exception as e:
            app.logger.error(f"Error in processing: {e}")
            result = "Error generating caption"
        result_queue.put(result)
        request_queue.task_done()

Thread(target=worker, daemon=True).start()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', model_size=model_size, quantized_model_size=quantized_model_size)

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    data = request.get_json(force=True)
    request_queue.put((model, data['image_data']))
    caption = result_queue.get()
    return jsonify(caption=caption)

@app.route('/generate_caption_quantized', methods=['POST'])
def generate_caption_quantized():
    data = request.get_json(force=True)
    request_queue.put((quantized_model, data['image_data']))
    caption = result_queue.get()
    return jsonify(caption=caption)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
