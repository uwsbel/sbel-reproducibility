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

for form in rp rA reps
do

    if [ "$allie_sim" = true ] && [ "$form" = "reps" ]; then
        continue
    fi

    for model in single_pendulum double_pendulum four_link slider_crank
    do

        for mode in kinematics dynamics
        do
            if [ "$model" = "double_pendulum" ] && [ "$mode" = "kinematics" ]; then
                continue
            fi

            if [ "$allie_sim" = true ]; then
                python3 ${model}_${mode}_${form}.py >> tmp.log
            else
                python3 ${model}.py --form $form --mode $mode --output tmp.log
            fi

            iters=$(grep 'Avg. iterations:' tmp.log | cut -d " " -f3)
            echo $iters
            echo "${form} ${model} ${mode}: ${iters}" >> iters.log

            rm -f tmp.log
        done
    done
done

rm -f tmp.log