# README

> Code Supplementary for *Unsupervised Hashing with Contrastive Learning by Exploiting Similarity Knowledge and Hidden Structure of Data*

## Settings

1. Downloaded datasets will be placed at `datasets/`.
2. Refer to `utils/data_path.py` to set dataset path.
3. Place pre-trained models at `models/pretrained_backbones/`.
4. Refer to `configs/globals.yml` to set names of pre-trained models.
5. Detail training configurations are set in `./configs/[dataset]/[stage]_[dataset]_[code_length].yml`.

## Training Example for 64-bit Cifar-10

```shell
chmod +x ./run.sh [CODE_LENGTH] [GPU_ID]
./run.sh
```
