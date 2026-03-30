#!/usr/bin/env bash
# scripts/deploy_ec2.sh
# Provisions an EC2 g4dn.xlarge, copies the app, and starts it with systemd.
# Prerequisites: AWS CLI configured, SSH key at $KEY_PATH.

set -euo pipefail

AMI_ID="ami-0f5ee92e2d63afc18"   # Deep Learning AMI (Ubuntu 22.04) ap-south-1
INSTANCE_TYPE="g4dn.xlarge"       # 1x T4 GPU, 16 GB VRAM
KEY_NAME="${KEY_NAME:-rag-llama3-key}"
KEY_PATH="${KEY_PATH:-~/.ssh/${KEY_NAME}.pem}"
SG_NAME="rag-llama3-sg"
APP_DIR="/opt/rag-llama3"

echo "▶ Creating security group …"
SG_ID=$(aws ec2 create-security-group \
  --group-name "$SG_NAME" \
  --description "RAG-LLaMA3 API" \
  --query 'GroupId' --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0 2>/dev/null || true

echo "▶ Launching EC2 instance …"
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-llama3}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "  Instance: $INSTANCE_ID"
echo "▶ Waiting for instance to start …"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "  Public IP: $PUBLIC_IP"
echo "▶ Uploading application …"
sleep 30  # wait for SSH daemon
rsync -avz --exclude='.git' --exclude='__pycache__' \
  -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" \
  ./ ubuntu@$PUBLIC_IP:$APP_DIR/

echo "▶ Running remote setup …"
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP bash << 'REMOTE'
  set -e
  cd /opt/rag-llama3
  python -m pip install -r requirements.txt --quiet

  # Create systemd service
  sudo tee /etc/systemd/system/rag-llama3.service > /dev/null << 'SERVICE'
[Unit]
Description=RAG-LLaMA3 API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/rag-llama3
EnvironmentFile=/opt/rag-llama3/.env
ExecStart=/usr/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE

  sudo systemctl daemon-reload
  sudo systemctl enable rag-llama3
  sudo systemctl start rag-llama3
  echo "✅ Service started"
REMOTE

echo ""
echo "✅ Deployment complete!"
echo "   API: http://$PUBLIC_IP:8000/docs"
echo "   Metrics: http://$PUBLIC_IP:8000/metrics"
