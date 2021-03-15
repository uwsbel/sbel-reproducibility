#!/usr/bin/env bash
#SBATCH --time 0-02:00:00
#SBATCH --job-name order-analysis
#SBATCH --nodelist euler20
#SBATCH --output rA-%j-%n.out
#SBATCH --error rA-%j-%n.err
#SBATCH --cpus-per-task 1
#SBATCH --partition wacc

module load anaconda
bootstrap_conda

python3 order_analysis.py