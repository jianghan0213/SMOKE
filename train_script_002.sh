CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/plain_train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:49852' --config-file configs/smoke_gn_vector_005nd.yaml
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_005nd.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/plain_train_net.py --num-gpus 4 --dist-url 'tcp://127.0.0.1:49852' --config-file configs/smoke_gn_vector_006nd.yaml
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_script.py --config-file configs/smoke_gn_vector_00n6d.yaml
