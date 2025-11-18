#!/bin/bash

python3 main_fnode.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.003     --lr_scheduler exponential     --lr_decay_rate 0.98   --optimizer  adam 

python3 main_fnode.py     --test_case Single_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98   --optimizer  adam

python3 main_fnode.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98  --optimizer  adam   --fnode_use_hybrid_target

python3 main_fnode.py     --test_case Cart_Pole     --seed 42     --data_total_steps 250     --train_ratio 0.8     --epochs 450     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98    --optimizer  adam     --fnode_use_hybrid_target

python3 main_fnode.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")    --epochs 450     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98    --optimizer  adam  --fnode_use_hybrid_target

python main_mbdnode.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98   

python3 main_mbdnode.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98   

python3 main_mbdnode.py     --test_case Single_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98     

python3 main_mbdnode.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98    

python3 main_lstm.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98 

python3 main_lstm.py     --test_case Single_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98   

python3 main_lstm.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98  

python3 main_lstm.py     --test_case Cart_Pole     --seed 42     --data_total_steps 250     --train_ratio 0.8     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98 

python main_lstm.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98 

python3 main_fcnn.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.003    --lr_scheduler exponential     --lr_decay_rate 0.98 

python3 main_fcnn.py     --test_case Single_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98   

python3 main_fcnn.py     --test_case Cart_Pole     --seed 42     --data_total_steps 250     --train_ratio 0.8     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98 

python main_fcnn.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98  

python3 main_fcnn.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.001    --lr_scheduler exponential  --lr_decay_rate 0.98