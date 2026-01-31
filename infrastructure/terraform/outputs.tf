# Terraform Outputs for LLM Inference Infrastructure

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.llm_inference.id
}

output "public_ip" {
  description = "Public IP address (Elastic IP)"
  value       = aws_eip.main.public_ip
}

output "public_dns" {
  description = "Public DNS name"
  value       = aws_eip.main.public_dns
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = "http://${aws_eip.main.public_ip}:8000"
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = "http://${aws_eip.main.public_ip}:3000"
}

output "prometheus_url" {
  description = "Prometheus URL"
  value       = "http://${aws_eip.main.public_ip}:9090"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i <your-key.pem> ubuntu@${aws_eip.main.public_ip}"
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "subnet_id" {
  description = "Public subnet ID"
  value       = aws_subnet.public.id
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.llm_inference.id
}

output "s3_bucket" {
  description = "S3 bucket for logs"
  value       = aws_s3_bucket.logs.bucket
}

output "ami_used" {
  description = "AMI ID used for the instance"
  value       = data.aws_ami.deep_learning.id
}

output "ami_name" {
  description = "AMI name used for the instance"
  value       = data.aws_ami.deep_learning.name
}
