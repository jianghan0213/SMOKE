CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/plain_train_net.py --num-gpus 4 --config-file configs/smoke_gn_vector.yaml --ckpt checkpoints/DLA-34-DCN_WEIGHTED_LOSS_005nd/model_0014000.pth
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_script.py --config-file configs/smoke_gn_vector.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/plain_train_net.py --num-gpus 4 --config-file configs/smoke_gn_vector_002nd.yaml
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_002nd.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/plain_train_net.py --num-gpus 4 --config-file configs/smoke_gn_vector_003nd.yaml
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_003nd.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/plain_train_net.py --num-gpus 4 --config-file configs/smoke_gn_vector_004nd.yaml
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_004nd.yaml