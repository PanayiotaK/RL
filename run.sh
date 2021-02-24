#!/bin/bash
# This line is required to inform the Linux
#command line to parse the script using
#the bash shell

# Instructing SLURM to locate and assign
#X number of nodes with Y number of
#cores in each node.
# X,Y are integers. Refer to table for
#various combinations
#SBATCH -N 1
#SBATCH -c 1

# Governs the run time limit and
# resource limit for the job. Please pick values
# from the partition and QOS tables below
#for various combinations
#SBATCH -p ug-gpu-small
#SBATCH --qos short
#SBATCH --gres=gpu:1
#SBATCH --mem=28g
#SBATCH -t 02-00:00:00
# Source the bash profile (required to use the module command)
source /etc/profile
#use cuda 
module load cuda/11.0-cudnn8.0
#virtual env
source  rl/bin/activate 
# Run your program (replace this with your program)
python3 RL/rl_1.py

