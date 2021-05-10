#!/bin/bash
#
#SBATCH --job-name=hybrid
#SBATCH --output=hybrid.log
#SBATCH --error=hybrid.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=huiying.li@sjsu.edu
#SBATCH --mail-type=END

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

module load python3/3.6.6
python3 -V
python3 /home/012289069/cmpe255/hybrid.py