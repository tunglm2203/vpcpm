CUDA_VISIBLE_DEVICES=3 python train.py \
    --domain_name jacob \
    --task_name reach_site_features \
    --encoder_type identity --work_dir ./tmp/reach_site_features \
    --action_repeat 4 --num_eval_episodes 10 \
    --pre_transform_image_size 84 --image_size 84 \
    --agent rad_sac --frame_stack 3   \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000 \
    --save_tb  --aug_coef 1 --add_str _w_ucb