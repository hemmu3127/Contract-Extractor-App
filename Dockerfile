# Stage 1: Build stage (optional, but good for cleaner final images if you have build steps)
# For this Python app, it's simpler to do it in one stage for now.

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables to ensure Python outputs everything directly (no buffering)
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install system dependencies that might be needed by some Python packages
# (e.g., build-essential for packages that compile C extensions, though ChromaDB might handle this)
# For python:3.9-slim, common ones are already there or handled by pip.
# If you face issues with pip install later, you might need to add some here:
# RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev etc. \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes the 'app' directory, '.env', and 'data' directory.
COPY ./app ./app
COPY .env .
COPY ./data ./data
# Note: The .env file copied here will be used if environment variables are not set
# directly in the docker run command or docker-compose.yml.
# For production, it's often better to pass secrets as environment variables.

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# The host 0.0.0.0 is important to make it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]