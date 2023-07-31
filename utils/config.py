import os

import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def create_config(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(
        pretext_dir, 'topk-train-neighbors.npy')

    # CGHash
    if cfg['setup'] == 'cghash':
        cghash_dir = os.path.join(base_dir, 'cghash')
        cghash_subdir = os.path.join(cghash_dir, cfg['cghash_subdir'])
        mkdir_if_missing(cghash_dir)
        mkdir_if_missing(cghash_subdir)
        cfg['cghash_dir'] = cghash_dir
        cfg['cghash_checkpoint'] = os.path.join(
            cghash_subdir, cfg['cghash_checkpoint'])
        cfg['cghash_model'] = os.path.join(cghash_subdir, cfg['cghash_model'])

    # Self-Labeling
    if cfg['setup'] == 'selflabel':
        cghash_dir = os.path.join(base_dir, 'cghash')
        cghash_subdir = os.path.join(cghash_dir, cfg['cghash_subdir'])
        assert os.path.exists(cghash_subdir), "cghash Model Not Exists."

        selflabel_dir = os.path.join(base_dir, 'selflabel')
        selflabel_subdir = os.path.join(selflabel_dir, cfg['selflabel_subdir'])
        mkdir_if_missing(selflabel_dir)
        mkdir_if_missing(selflabel_subdir)
        cfg['cghash_dir'] = cghash_dir
        cfg['selflabel_dir'] = selflabel_dir
        cfg['cghash_checkpoint'] = os.path.join(
            cghash_subdir, cfg['cghash_checkpoint'])
        cfg['cghash_model'] = os.path.join(cghash_subdir, cfg['cghash_model'])
        cfg['selflabel_checkpoint'] = os.path.join(
            selflabel_subdir, cfg['selflabel_checkpoint'])
        cfg['selflabel_model'] = os.path.join(
            selflabel_subdir, cfg['selflabel_model'])

    return cfg
