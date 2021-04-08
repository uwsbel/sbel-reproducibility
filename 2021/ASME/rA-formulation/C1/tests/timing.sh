#!/usr/bin/env bash
#SBATCH --job-name multi-body-timing
#SBATCH --nodelist euler20
#SBATCH --output rA-%j-%n.out
#SBATCH --error rA-%j-%n.err
#SBATCH --cpus-per-task 1
#SBATCH --partition wacc

module load anaconda
bootstrap_conda

allie_sim=true

for form in rA rp reps
do
    for model in single_pendulum double_pendulum four_link slider_crank
    do

        for mode in kinematics dynamics
        do

            if [ "$model" = "double_pendulum" ] && [ "$mode" = "kinematics" ]; then
                continue
            fi

            tmp_file=tmp.log

            for i in {1..10}
            do
                if [ "$allie_sim" = true ]; then
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