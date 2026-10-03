# Use a base image with Python
FROM python:3.9-slim

# Copy your model file to the container
# Replace with your actual model file name
COPY your-model.gguf /app/model.gguf

# Copy your inference script to the container
COPY inference.py /app/inference.py

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio transformers accelerate sentencepiece

# Set the working directory
WORKDIR /app

# Command to run the inference script
CMD ["python", "inference.py"]