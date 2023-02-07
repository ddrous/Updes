#!/bin/bash


#### Set-up ressources #####

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job
#SBATCH --job-name=updec

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
##SBATCH --mail-type=ALL

# send mail to this address
##SBATCH --mail-user=gb21553.brown@gmail.com

# Set the folder for output
#SBATCH --output ./scripts/reports/%j.out



#### run the application #####

## Load the tensorflow GPU container
/jmain02/apps/docker/tensorflow -c

## Activate Conda (Make sure dependencies are in-there)
source activate base
conda activate jaxenv

## Run Python script
python3 ./examples/01_laplace_with_rbf.py
