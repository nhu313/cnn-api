# Use the official Ubuntu 22.04 image as the base
FROM python:3.10-slim

# Set environment variables to prevent interactive prompts
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Etc/UTC

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .


# Install any needed Python packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install opencv-python


# Copy the current directory contents into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable
ENV FLASK_APP=main.py

# Use a production WSGI server (e.g., Gunicorn)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "main:app"]
# CMD "flask run --host=0.0.0.0 --port=8080"
