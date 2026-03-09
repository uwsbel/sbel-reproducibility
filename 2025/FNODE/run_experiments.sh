#!/bin/bash

# # ========== Single Mass Spring experiments ==========
# HNN (for reference - already exists)
python3 main_HNN.py --test_case Single_Mass_Spring --dt 0.01 --t_end 30.0 --num_epochs 30000 --lr 1e-3 --num_steps_test 3000

# LNN (for reference - already exists)
python3 main_LNN.py --test_case Single_Mass_Spring --dt 0.01 --num_steps 3000 --training_size 300 --num_epochs 400 --lr 1e-4 --num_steps_test 3000

# FNODE for Single Mass Spring
python3 main_fnode.py --test_case Single_Mass_Spring --seed 42 --data_total_steps 3000 --train_ratio 0.1 --epochs 400 --lr 0.001 --lr_scheduler exponential --lr_decay_rate 0.98 --optimizer adam

# MBDNODE for Single Mass Spring
python3 main_mbdnode.py --test_case Single_Mass_Spring --seed 42 --data_total_steps 3000 --train_ratio 0.1 --epochs 300 --lr 0.001 --lr_scheduler exponential --lr_decay_rate 0.98 --numerical_methods yoshida4

# LSTM for Single Mass Spring
python3 main_lstm.py --test_case Single_Mass_Spring --seed 42 --data_total_steps 3000 --train_ratio 0.1 --epochs 300 --lr 0.001 --lr_scheduler exponential --lr_decay_rate 0.98 --lstm_seq_len 16

# FCNN for Single Mass Spring
python3 main_fcnn.py --test_case Single_Mass_Spring --seed 42 --data_total_steps 3000 --train_ratio 0.1 --epochs 400 --lr 0.001 --lr_scheduler exponential --lr_decay_rate 0.98

# ========== Other experiments ==========
python3 main_fnode.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.003     --lr_scheduler exponential     --lr_decay_rate 0.98   --optimizer  adam 

python3 main_fnode.py     --test_case Single_Mass_Spring_Damper     --seed 42   --fnode_accel_mtd analytical  --data_total_steps 8000     --train_ratio 0.05     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98   --optimizer  adam --data_generation_method analytical --data_dt 0.005

python3 main_fnode.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98  --optimizer  adam 

python3 main_fnode.py     --test_case Cart_Pole     --seed 42     --data_total_steps 250     --train_ratio 0.8     --epochs 450     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98    --optimizer  adam     

python3 main_fnode.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")    --epochs 450     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98    --optimizer  adam  

python main_mbdnode.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98   

python3 main_mbdnode.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98   

python3 main_mbdnode.py     --test_case Single_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98     

python3 main_mbdnode.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98    

python3 main_mbdnode.py     --test_case Cart_Pole     --seed 42     --data_total_steps 250     --train_ratio 0.8     --epochs 300     --lr 0.001     --lr_scheduler  exponential     --lr_decay_rate 0.98    

python3 main_lstm.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98 

python3 main_lstm.py     --test_case Single_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98   

python3 main_lstm.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98  

python3 main_lstm.py     --test_case Cart_Pole     --seed 42     --data_total_steps 250     --train_ratio 0.8     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98

# ========== Vehicle 4DOF with Control ==========
# FNODE for 4DOF Vehicle with control inputs
python3 main_fnode_veh.py --test_case veh_4dof --seed 42 --data_dt 0.01 --data_total_steps 4000 --train_ratio 0.75 --epochs 300 --lr 0.001 --lr_scheduler exponential --lr_decay_rate 0.98 --optimizer adam --layers 3 --hidden_size 128 --activation tanh

python main_lstm.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")     --epochs 300     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98 

python3 main_fcnn.py     --test_case Double_Pendulum     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.003    --lr_scheduler exponential     --lr_decay_rate 0.98 

python3 main_fcnn.py     --test_case Single_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98   

python3 main_fcnn.py     --test_case Cart_Pole     --seed 42     --data_total_steps 250     --train_ratio 0.8     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98 

python main_fcnn.py     --test_case Slider_Crank     --seed 42     --data_total_steps 4500     --train_ratio $(python3 -c "print(1/3)")     --epochs 450     --lr 0.001     --lr_scheduler exponential     --lr_decay_rate 0.98  

python3 main_fcnn.py     --test_case Triple_Mass_Spring_Damper     --seed 42     --data_total_steps 400     --train_ratio 0.75     --epochs 450     --lr 0.001    --lr_scheduler exponential  --lr_decay_rate 0.98


python main_fnode.py \
    --test_case veh_11dof \
    --data_dt 0.01 \
    --data_total_steps 1000 \
    --data_train_split 0.7 \
    --epochs 3 \
    --layers 3 \
    --hidden_size 256 \
    --lr 0.001 \
    --activation tanh \
    --ode_method rk4 \
    --out_time_log 1

python main_fnode_veh11dof.py \
    --epochs 400 \
    --data_total_steps 5000 \
    --train_ratio 0.5 \
    --hidden_size 512 \
    --layers 3 \
    --lr 0.001 \
    --batch_size 32