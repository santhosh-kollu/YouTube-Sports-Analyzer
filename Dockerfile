# Use an official Python runtime as a parent image
FROM python:3.10-slim

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

# Overwrite PyPI yt-dlp with the bleeding-edge master branch to bypass YouTube blocks
# Also install curl_cffi for TLS fingerprint impersonation (pinned to 0.5.9 to prevent OpenSSL crashing bug on Debian)
RUN pip install -U https://github.com/yt-dlp/yt-dlp/archive/master.zip "curl_cffi==0.5.9"

# Copy the rest of the application
COPY . .

# Create data directory with appropriate permissions
RUN mkdir -p data && chmod 777 data

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the application
CMD ["python", "run.py"]
