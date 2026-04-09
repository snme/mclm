#!/bin/bash

# Get the head node (first node in the allocation)
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"

# Start Ray on the head node
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --node-ip-address="$HEAD_NODE_IP" --port=6379 --block &

sleep 10  # Wait for head to initialize

# Start Ray workers on all OTHER nodes
srun --nodes=9 --ntasks=9 --exclude="$HEAD_NODE" \
    ray start --address="$HEAD_NODE_IP:6379" --block &

sleep 10  # Wait for workers to join

echo "Ray cluster ready. Running job..."
