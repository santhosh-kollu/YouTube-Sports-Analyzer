# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies (ffmpeg is required by yt-dlp and whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory with appropriate permissions
RUN mkdir -p data && chmod 777 data

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the application
CMD ["python", "run.py"]
