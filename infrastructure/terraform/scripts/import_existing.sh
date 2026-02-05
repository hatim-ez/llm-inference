#!/usr/bin/env bash
set -euo pipefail

# Auto-discover existing AWS resources and import them into the current Terraform state.
# Defaults follow variable defaults in main.tf; override with env vars:
#   PROJECT_NAME, AWS_REGION, AWS_PROFILE
# Optional manual overrides if discovery fails:
#   VPC_ID, SUBNET_ID, IGW_ID, RTB_ID, SG_ID
# The script looks up resources by Name and Project tags to survive slight naming drift.
#
# Prereqs: AWS CLI v2, Terraform CLI, jq (for readable output).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"   # infrastructure/terraform
PROJECT_NAME="${PROJECT_NAME:-llm-inference}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-default}"

AWS_ARGS=(--region "$AWS_REGION")
[[ -n "$AWS_PROFILE" ]] && AWS_ARGS+=(--profile "$AWS_PROFILE")

cd "$TF_DIR"

echo "Project: $PROJECT_NAME"
echo "Region : $AWS_REGION"
[[ -n "$AWS_PROFILE" ]] && echo "Profile: $AWS_PROFILE"

account_id() {
  aws "${AWS_ARGS[@]}" sts get-caller-identity --query Account --output text
}

first_nonempty() { # pick the first nonempty argument
  for v in "$@"; do [[ -n "$v" && "$v" != "None" && "$v" != "null" ]] && { echo "$v"; return; }; done
}

find_vpc() {
  local tag_name="${PROJECT_NAME}-vpc"
  local by_name by_project
  by_name=$(aws "${AWS_ARGS[@]}" ec2 describe-vpcs \
    --filters "Name=tag:Name,Values=${tag_name}" \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null || true)
  by_project=$(aws "${AWS_ARGS[@]}" ec2 describe-vpcs \
    --filters "Name=tag:Project,Values=${PROJECT_NAME}" \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null || true)
  first_nonempty "$by_name" "$by_project"
}

find_subnet() {
  local tag_name="${PROJECT_NAME}-public-subnet"
  aws "${AWS_ARGS[@]}" ec2 describe-subnets \
    --filters "Name=tag:Name,Values=${tag_name}" \
              "Name=tag:Project,Values=${PROJECT_NAME}" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null || true
}

find_rt() {
  local vpc_id=$1
  aws "${AWS_ARGS[@]}" ec2 describe-route-tables \
    --filters "Name=vpc-id,Values=${vpc_id}" \
              "Name=tag:Name,Values=${PROJECT_NAME}-public-rt" \
    --query 'RouteTables[0].RouteTableId' --output text 2>/dev/null || true
}

find_sg() {
  aws "${AWS_ARGS[@]}" ec2 describe-security-groups \
    --filters "Name=tag:Name,Values=${PROJECT_NAME}-sg" \
              "Name=tag:Project,Values=${PROJECT_NAME}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true
}

import_if_found() {
  local addr=$1 id=$2
  if [[ -n "$id" && "$id" != "None" && "$id" != "null" ]]; then
    echo "Importing $addr <- $id"
    terraform import "$addr" "$id"
  else
    echo "Skipping $addr (not found)"
  fi
}

# # --- VPC + networking ---
# VPC_ID="$(find_vpc)"
# import_if_found aws_vpc.main "${VPC_ID:-${VPC_ID_OVERRIDE:-}}"

# SUBNET_ID="${SUBNET_ID_OVERRIDE:-$(find_subnet)}"
# import_if_found aws_subnet.public "$SUBNET_ID"

# ACTIVE_VPC="${VPC_ID_OVERRIDE:-$VPC_ID}"
# if [[ -n "$ACTIVE_VPC" && "$ACTIVE_VPC" != "None" ]]; then
#   IGW_ID="${IGW_ID_OVERRIDE:-$(aws "${AWS_ARGS[@]}" ec2 describe-internet-gateways \
#     --filters "Name=attachment.vpc-id,Values=${ACTIVE_VPC}" \
#     --query 'InternetGateways[0].InternetGatewayId' --output text 2>/dev/null || true)}"
#   import_if_found aws_internet_gateway.main "$IGW_ID"

#   RTB_ID="${RTB_ID_OVERRIDE:-$(find_rt "$ACTIVE_VPC")}"
#   import_if_found aws_route_table.public "$RTB_ID"

#   SG_ID="${SG_ID_OVERRIDE:-$(find_sg)}"
#   import_if_found aws_security_group.llm_inference "$SG_ID"
# fi

# # --- IAM for EC2 ---
# import_if_found aws_iam_role.ec2_role "${PROJECT_NAME}-ec2-role"
# import_if_found aws_iam_role_policy.ec2_policy "${PROJECT_NAME}-ec2-role:${PROJECT_NAME}-ec2-policy"
import_if_found aws_iam_instance_profile.ec2_profile "${PROJECT_NAME}-ec2-profile"

# # --- Budgets + actions ---
# import_if_found aws_iam_role.budget_actions "${PROJECT_NAME}-budget-actions"
# import_if_found aws_iam_role_policy.budget_actions "${PROJECT_NAME}-budget-actions:${PROJECT_NAME}-budget-actions-policy"

# BUDGET_NAME="${PROJECT_NAME}-per-month-budget"
# ACCOUNT_ID="$(account_id)"
# import_if_found aws_budgets_budget.monthly "${ACCOUNT_ID}:${BUDGET_NAME}"

echo "Imports attempted. Review above output, then run: terraform plan"
