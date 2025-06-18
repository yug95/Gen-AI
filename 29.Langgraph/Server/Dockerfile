# Start from a lightweight Python base image
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Copy your project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn[standard]

# Expose the port
EXPOSE 8000

# Start the app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]