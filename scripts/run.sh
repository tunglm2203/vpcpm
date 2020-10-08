#!/usr/bin/env bash
# TODO: For state
#CUDA_VISIBLE_DEVICES=0 python train.py \
#    --domain_name cartpole \
#    --task_name swingup \
#    --encoder_type identity --work_dir ./logs \
#    --action_repeat 1 --num_eval_episodes 10 \
#    --pre_transform_image_size 100 --image_size 84 \
#    --agent rad_sac --frame_stack 1  \
#    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --critic_tau 0.01\
#    --eval_freq 10000 --batch_size 128 --num_train_steps 200000\
#    --save_tb

# Same with rlkit
#ENV_NAME='reach_target-state-v0'
#EXP_NAME='baseline-200k'
#CUDA_VISIBLE_DEVICES=0 python train_rlbench.py \
#    --domain_name panda \
#    --task_name $ENV_NAME \
#    --encoder_type identity --work_dir ./logs \
#    --action_repeat 1 --num_eval_episodes 10 \
#    --agent rad_sac --frame_stack 1 \
#    --seed 1 \
#    --exp $EXP_NAME \
#    --critic_beta 0.9 --critic_lr 3e-4 --actor_beta 0.9 --actor_lr 3e-4 --critic_tau 0.001 \
#    --alpha_lr 3e-4 --alpha_beta 0.9 \
#    --actor_log_std_min -20 --actor_log_std_max 2 \
#    --critic_target_update_freq 1 --actor_update_freq 1 \
#    --eval_freq 1000 --batch_size 256 --num_train_steps 200000 --init_steps 1000 \
#    --training_freq 500 --num_updates 50 \
#    --save_tb --save_model

# TODO:  For image
#CUDA_VISIBLE_DEVICES=1 python train.py \
#    --domain_name cartpole \
#    --task_name swingup \
#    --encoder_type pixel --work_dir ./logs \
#    --action_repeat 8 --num_eval_episodes 10 \
#    --pre_transform_image_size 100 --image_size 84 \
#    --agent rad_sac --frame_stack 3 --data_augs crop  \
#    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000\
#    --save_tb

ENV_NAME='reach_target-vision-v0'
EXP_NAME='baseline-200k-stack1-update50-simple-env-depth-crop'
CUDA_VISIBLE_DEVICES=0 python train_rlbench.py \
    --domain_name panda \
    --task_name $ENV_NAME \
    --encoder_type pixel --work_dir ./logs \
    --action_repeat 1 --num_eval_episodes 10 \
    --agent rad_sac --frame_stack 1 --data_augs crop \
    --seed 1 \
    --critic_beta 0.9 --critic_lr 3e-4 --actor_beta 0.9 --actor_lr 3e-4 --critic_tau 0.001 \
    --alpha_lr 3e-4 --alpha_beta 0.9 \
    --actor_log_std_min -20 --actor_log_std_max 2 \
    --exp $EXP_NAME \
    --critic_target_update_freq 1 --actor_update_freq 1 \
    --eval_freq 1000 --batch_size 256 --num_train_steps 200000 --init_steps 1000 \
    --training_freq 500 --num_updates 50 \
    --save_tb --save_model --use_depth
