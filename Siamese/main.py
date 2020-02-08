import torch
import random

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config, load_config
from data_loader import get_train_valid_loader, get_test_loader


def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # create Omniglot data loaders
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}
    if config.is_train:
        data_loader = get_train_valid_loader(
            config.data_dir, config.batch_size,
            config.num_train, config.augment,
            config.way, config.valid_trials,
            config.shuffle, config.random_seed,
            **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.way,
            config.test_trials, config.random_seed,
            **kwargs
        )

    # sample 3 layer wise hyperparams if first time training
    if config.is_train and not config.resume:
        print("[*] Sampling layer hyperparameters.")

        layer_hyperparams = {
            'layer_init_lrs': [],
            'layer_end_momentums': [],
            'layer_l2_regs': []
        }
        for i in range(6):
            # sample
            lr = random.uniform(1e-4, 1e-1)
            mom = random.uniform(0, 1)
            reg = random.uniform(0, 0.1)

            # store
            layer_hyperparams['layer_init_lrs'].append(lr)
            layer_hyperparams['layer_end_momentums'].append(mom)
            layer_hyperparams['layer_l2_regs'].append(reg)
        try:
            save_config(config, layer_hyperparams)
        except ValueError:
            print(
                "[!] Samples already exist. Either change the model number,",
                "or delete the json file and rerun.",
                sep=' '
            )
            return
    # else load it from config file
    else:
        try:
            layer_hyperparams = load_config(config)
        except FileNotFoundError:
            print("[!] No previously saved config. Set resume to False.")
            return

    trainer = Trainer(config, data_loader, layer_hyperparams)

    if config.is_train:
        trainer.train()
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
