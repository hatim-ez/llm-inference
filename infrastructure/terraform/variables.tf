# Terraform Variables for LLM Inference Infrastructure

# Project Configuration
variable "project_name" {
  description = "Name of the project (used for resource naming)"
  type        = string
  default     = "llm-inference"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

# Security Configuration
variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = []  # Must be set explicitly for security
}

variable "allowed_api_cidrs" {
  description = "CIDR blocks allowed for direct API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Open by default, restrict in production
}

variable "allowed_monitoring_cidrs" {
  description = "CIDR blocks allowed for monitoring endpoints"
  type        = list(string)
  default     = []  # Must be set explicitly
}

# SSH Key Configuration
variable "ssh_public_key" {
  description = "SSH public key to create new key pair (leave empty to use existing)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "existing_key_name" {
  description = "Name of existing SSH key pair (used if ssh_public_key is empty)"
  type        = string
  default     = ""
}

# EC2 Configuration
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g5.2xlarge"  # 1x NVIDIA A10G 24GB
}

variable "root_volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 500
}

# Model Configuration
variable "model_name" {
  description = "Name/path of the LLM model"
  type        = string
  default     = "meta-llama/Llama-3.2-11B-Vision-Instruct"
}

variable "gpu_memory_utilization" {
  description = "GPU memory utilization for vLLM (0.0-1.0)"
  type        = number
  default     = 0.90
}

# Cost Controls
variable "budget_limit" {
  description = "Monthly cost budget in USD"
  type        = number
  default     = 50
}

variable "budget_alert_threshold" {
  description = "Percentage of budget that triggers alert emails"
  type        = number
  default     = 80
}

variable "budget_alert_email" {
  description = "Email address to receive budget alerts"
  type        = string
  default     = "replace-me@example.com"
}

variable "enable_budget_action" {
  description = "Whether to auto-stop the EC2 instance when budget threshold is crossed"
  type        = bool
  default     = false
}

variable "budget_action_threshold" {
  description = "Percentage of budget that triggers the stop action"
  type        = number
  default     = 100
}
