#!/bin/bash

#SBATCH -o mapsim_%j.out
#SBATCH -e mapsim_%J.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=scottgbarnes@gwu.edu

#SBATCH -N 1
#SBATCH -p cpu

#SBATCH -D /home/scottgbarnes/map_sim
#SBATCH -J mapsim

#SBATCH -t 24:00:00

module load anaconda/4.4.6
python3 run_mapEnv.py
