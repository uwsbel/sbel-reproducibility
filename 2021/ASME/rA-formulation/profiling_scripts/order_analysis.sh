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

c1_sim=false

for form in rA rp reps
do
    if [ "$c1_sim" = true ] && [ "$form" = "reps" ]; then
        continue
    fi

    for model in single_pendulum four_link slider_crank
    do

        if [ "$c1_sim" = true ] && [ "$model" = "single_pendulum" ]; then
            continue
        fi
        
        tmp_file=tmp.log
        
        if [ "$c1_sim" = true ]; then
            python3 ${model}_kinematics_${form}.py >> $tmp_file
        else
            python3 ${model}.py --form $form --mode kin --tol 1e-12 --output $tmp_file --save_data
        fi

        echo "${form} ${model} ${mode}" >> order.log
        for steps in 1e-2 1e-3 1e-4
        do
            python3 ${model}.py --form $form --mode dyn --step_size $steps --output $tmp_file --read_data
            echo $steps >> order.log
            grep 'diff:' $tmp_file | cut -d " " -f 3- >> order.log
            rm -f $tmp_file 
        done
        rm -f *.csv

        echo "Finished ${form} ${model} ${mode}" >> progress.log

    done
done