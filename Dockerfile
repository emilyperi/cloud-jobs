# Use the Vertex AI base image
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest

# Install TensorFlow 2.13.1
RUN pip install tensorflow==2.15.0

# Set the default command to execute the training script
ENTRYPOINT ["/launcher.sh", "python", "/runcloudml.py"]