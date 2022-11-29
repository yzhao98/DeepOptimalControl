python -u gen_fix.py --data_type='train' --data_size=500 --bvp_type='space_march_fw' --seed=1 --problem_id=2_4
python -u gen_fix.py --data_type='train' --data_size=500 --bvp_type='space_march_fw' --seed=1 --problem_id=2_8
python -u gen_fix.py --data_type='train' --data_size=500 --bvp_type='space_march_fw' --seed=1 --problem_id=2_16
python -u gen_fix.py --data_type='train' --data_size=1000 --bvp_type='space_march_fw' --seed=1 --problem_id=10_16


python -u gen_fix.py --data_type='valid' --data_size=100 --bvp_type='space_march_fw' --seed=0 --problem_id=2_4
python -u gen_fix.py --data_type='valid' --data_size=100 --bvp_type='space_march_fw' --seed=0 --problem_id=2_8
python -u gen_fix.py --data_type='valid' --data_size=100 --bvp_type='space_march_fw' --seed=0 --problem_id=2_16
python -u gen_fix.py --data_type='valid' --data_size=100 --bvp_type='space_march_fw' --seed=0 --problem_id=10_16