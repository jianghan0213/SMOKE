g=$(($2<8?$2:8))
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --cpus-per-task=4 --ntasks-per-node=$g \
python tools/plain_train_net.py --num-gpus 4 --config-file configs/smoke_gn_vector.yaml

srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --cpus-per-task=4 --ntasks-per-node=$g \
python tools/evaluate_script.py --config-file configs/smoke_gn_vector.yaml