#!/bin/bash
# User data script for LLM Inference EC2 instance
set -e

# Log everything
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting user data script at $(date)"

# Variables from Terraform
PROJECT_NAME="${project_name}"
S3_BUCKET="${s3_bucket}"
AWS_REGION="${aws_region}"
MODEL_NAME="${model_name}"
GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}"

# System updates
echo "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install required packages
echo "Installing required packages..."
apt-get install -y \
    git \
    curl \
    wget \
    htop \
    nvtop \
    jq \
    docker.io \
    docker-compose \
    nginx \
    certbot \
    python3-certbot-nginx

# Enable Docker
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# Create directories
echo "Creating project directories..."
mkdir -p /home/ubuntu/llm-inference
mkdir -p /home/ubuntu/models
mkdir -p /home/ubuntu/logs
mkdir -p /home/ubuntu/monitoring

# Set ownership
chown -R ubuntu:ubuntu /home/ubuntu

# Clone project repository (if using git)
# git clone https://github.com/yourusername/llm-inference.git /home/ubuntu/llm-inference

# Create virtual environment
echo "Setting up Python environment..."
cd /home/ubuntu/llm-inference
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt || echo "requirements.txt not found, skipping"

# Create environment file
cat > /home/ubuntu/llm-inference/.env << EOF
MODEL_NAME=$MODEL_NAME
MODEL_PATH=/home/ubuntu/models/llama-3.2-11b-vision
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_MODEL_LENGTH=4096
MAX_NUM_SEQS=128
API_HOST=0.0.0.0
API_PORT=8000
RATE_LIMIT=100
REQUEST_TIMEOUT=300
REQUIRE_API_KEY=false
ENABLE_METRICS=true
LOG_LEVEL=INFO
AWS_REGION=$AWS_REGION
S3_BUCKET=$S3_BUCKET
EOF

# Create systemd service for the API
echo "Creating systemd service..."
cat > /etc/systemd/system/llm-api.service << EOF
[Unit]
Description=LLM Inference API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/llm-inference
Environment=PATH=/home/ubuntu/llm-inference/venv/bin
ExecStart=/home/ubuntu/llm-inference/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable but don't start (model needs to be downloaded first)
systemctl daemon-reload
systemctl enable llm-api

# Configure Nginx
echo "Configuring Nginx..."
cat > /etc/nginx/sites-available/llm-inference << EOF
upstream llm_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://llm_api;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /health {
        proxy_pass http://llm_api/health;
        access_log off;
    }

    location /metrics {
        proxy_pass http://llm_api/metrics;
        access_log off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/llm-inference /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
systemctl restart nginx

# Setup monitoring
echo "Setting up monitoring..."
cd /home/ubuntu/monitoring

# Create docker-compose for monitoring stack
cat > docker-compose.yml << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped

  dcgm-exporter:
    image: nvidia/dcgm-exporter:latest
    container_name: dcgm-exporter
    runtime: nvidia
    ports:
      - "9400:9400"
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF

# Create Prometheus config
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'llm-api'
    static_configs:
      - targets: ['host.docker.internal:8000']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'dcgm-exporter'
    static_configs:
      - targets: ['dcgm-exporter:9400']
EOF

# Start monitoring stack
docker-compose up -d || echo "Docker compose failed, may need manual setup"

# Set ownership
chown -R ubuntu:ubuntu /home/ubuntu

# Print completion message
echo "User data script completed at $(date)"
echo ""
echo "==============================================="
echo "Setup complete!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "1. SSH into the instance"
echo "2. Download the model:"
echo "   huggingface-cli login"
echo "   huggingface-cli download $MODEL_NAME --local-dir /home/ubuntu/models/llama-3.2-11b-vision"
echo "3. Start the API:"
echo "   sudo systemctl start llm-api"
echo "4. Check the health:"
echo "   curl http://localhost:8000/health"
echo ""
echo "Access:"
echo "- API: http://<public-ip>:8000"
echo "- Grafana: http://<public-ip>:3000 (admin/admin)"
echo "- Prometheus: http://<public-ip>:9090"
echo "==============================================="
