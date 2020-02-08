import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='Siamese Network')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_trials', type=int, default=320,
                      help='# of validation 1-shot trials')
data_arg.add_argument('--test_trials', type=int, default=400,
                      help='# of test 1-shot trials')
data_arg.add_argument('--way', type=int, default=20,
                      help='Ways in the 1-shot trials')
data_arg.add_argument('--num_train', type=int, default=90000,
                      help='# of images in train dataset')
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=1,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the dataset between epochs')
data_arg.add_argument('--augment', type=str2bool, default=True,
                      help='Whether to use data augmentation for train data')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--init_momentum', type=float, default=0.5,
                       help='Initial layer-wise momentum value')
train_arg.add_argument('--lr_patience', type=int, default=1,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=20,
                       help='Number of epochs to wait before stopping train')


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--flush', type=str2bool, default=False,
                      help='Whether to delete ckpt + log files for model no.')
misc_arg.add_argument('--num_model', type=int, default=1, required=True,
                      help='Model number used for unique checkpointing')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/changed/',
                      help='Directory in which data is stored')
misc_arg.add_argument('--plot_dir', type=str, default='./plots/',
                      help='Directory in which plots are stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
