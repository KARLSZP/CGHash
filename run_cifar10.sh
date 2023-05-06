# ./run.sh [CODE_LENGTH] [GPU_ID]
python mine_neighbors.py --config_env configs/env.yml --config_exp configs/cifar-10/simclr_cifar10.yml -g $3
python cghash.py --config_env configs/env.yml --config_exp configs/cifar-10/cghash_cifar_$2.yml -f -g $3
python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar_$2.yml -f -g $3