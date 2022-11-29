CUDA_VISIBLE_DEVICES=0 python train_sl.py --experiment_name 'satellite_fix' --problem_id "1_20" --num_epochs 200 --batch_size 1024 --lr 0.01 --use_gpu;

CUDA_VISIBLE_DEVICES=0 python train_sl.py --experiment_name 'quadrotor_fix' --problem_id "2_4" --lr 0.001 --num_epochs 1000  --batch_size 4096 --use_gpu;
CUDA_VISIBLE_DEVICES=0 python train_sl.py --experiment_name 'quadrotor_fix' --problem_id "2_8" --lr 0.001 --num_epochs 1000  --batch_size 4096 --use_gpu;
CUDA_VISIBLE_DEVICES=0 python train_sl.py --experiment_name 'quadrotor_fix' --problem_id "2_16" --lr 0.001 --num_epochs 1000  --batch_size 4096 --use_gpu;
CUDA_VISIBLE_DEVICES=0 python train_sl.py --experiment_name 'quadrotor_fix' --problem_id "10_16" --lr 0.001 --num_epochs 2000  --batch_size 4096 --use_gpu;
CUDA_VISIBLE_DEVICES=0 python train_sl.py --experiment_name 'quadrotor_adaptive' --problem_id "10_16" --lr 0.005 --num_epochs 2000  --batch_size 4096 --use_gpu;