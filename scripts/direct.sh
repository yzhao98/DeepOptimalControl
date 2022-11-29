CUDA_VISIBLE_DEVICES=0 python train_direct_qua.py --experiment_name 'quadrotor_fix' --lr 0.01 --X0_scale 2 --T 4 --batch_size 2048 --x_dim 12 --u_dim 4 --hidden_size 64 --seed 1 --num_iters 3000 --use_gpu;
CUDA_VISIBLE_DEVICES=0 python train_direct_qua.py --experiment_name 'quadrotor_fix' --lr 0.01 --X0_scale 2 --T 8 --batch_size 2048 --x_dim 12 --u_dim 4 --hidden_size 64 --seed 1 --num_iters 3000 --use_gpu;
CUDA_VISIBLE_DEVICES=0 python train_direct_qua.py --experiment_name 'quadrotor_fix' --lr 0.01 --X0_scale 2 --T 16 --batch_size 2048 --x_dim 12 --u_dim 4 --hidden_size 64 --seed 1 --num_iters 3000 --use_gpu;
CUDA_VISIBLE_DEVICES=0 python train_direct_qua.py --experiment_name 'quadrotor_fix' --lr 0.01 --X0_scale 10 --T 16 --batch_size 2048 --x_dim 12 --u_dim 4 --hidden_size 64 --seed 1 --num_iters 3000 --use_gpu;

CUDA_VISIBLE_DEVICES=1 python train_direct_sat.py --experiment_name 'satellite_fix' --lr 0.01 --X0_scale 1 --T 20 --batch_size 1024 --x_dim 6 --u_dim 3 --hidden_size 64 --seed 1 --num_iters 2000 --use_gpu
