CUDA_VISIBLE_DEVICES=4,5 python tools/plain_train_net.py --num-gpus 2 --config-file configs/smoke_gn_vector.yaml
CUDA_VISIBLE_DEVICES=4 python tools/evaluate_script.py --config-file configs/smoke_gn_vector.yaml

CUDA_VISIBLE_DEVICES=4,5 python tools/plain_train_net.py --num-gpus 2  --config-file configs/smoke_gn_vector_002nd.yaml
CUDA_VISIBLE_DEVICES=4 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_002nd.yaml

CUDA_VISIBLE_DEVICES=4,5 python tools/plain_train_net.py --num-gpus 2 --config-file configs/smoke_gn_vector_003nd.yaml
CUDA_VISIBLE_DEVICES=4 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_003nd.yaml

CUDA_VISIBLE_DEVICES=4,5 python tools/plain_train_net.py --num-gpus 2 --config-file configs/smoke_gn_vector_004nd.yaml
CUDA_VISIBLE_DEVICES=4 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_004nd.yaml