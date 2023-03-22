FROM python:3.9.13-slim-buster
# Set working directory
WORKDIR /app
# Copy application files
COPY config /app/config
COPY scripts /app/scripts
COPY requirements.txt /app
COPY main.py /app
# Install dependencies
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Define command to run the application
ENTRYPOINT [ "python", "main.py" ]