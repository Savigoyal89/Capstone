#!/bin/bash
#SBATCH --job-name=test
#SBATCH --error=test.err
#SBATCH --ntasks=64
#SBATCH --partition=debug

# Load the default OpenMPI module.
#module load openmpi
# Run the hellompi program with mpirun. The -n flag is not required;
# mpirun will automatically figure out the best configuration from the
# Slurm environment variables.

mpirun ./get_subsets