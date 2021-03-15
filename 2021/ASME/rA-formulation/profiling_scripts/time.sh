#!/usr/bin/env bash
#SBATCH --time 0-04:00:00
#SBATCH --job-name multi-body-timing
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

    for model in single_pendulum double_pendulum four_link slider_crank
    do

        if [ "$c1_sim" = true ] && [ "$model" = "single_pendulum" ]; then
            continue
        fi
        
        for mode in kinematics dynamics
        do

            if [ "$model" = "double_pendulum" ] && [ "$mode" = "kinematics" ]; then
                continue
            fi

            tmp_file=tmp.log
            
            for i in {1..20}
            do
                if [ "$c1_sim" = true ]; then
                    python3 ${model}_${mode}_${form}.py >> $tmp_file
                else
                    python3 ${model}.py --form $form --mode $mode --output $tmp_file
                fi
            done

            echo "${form} ${model} ${mode}" >> times.log
            grep 'Simulation time:' $tmp_file | cut -d " " -f3 >> times.log
            echo "Finished ${form} ${model} ${mode}" >> progress.log

            rm -f $tmp_file
        done
    done
done