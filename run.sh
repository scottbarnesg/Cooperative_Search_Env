#!/bin/bash

#SBATCH -o mapsim_%j.out
#SBATCH -e mapsim_%J.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=scottgbarnes@gwu.edu

#SBATCH -N 1
#SBATCH -p short

#SBATCH -D /home/scottgbarnes/Cooperative-Search-Gym
#SBATCH -J mapsim

#SBATCH -t 2-00:00:00

module load anaconda
source activate tf3
python run_mapEnv.py
