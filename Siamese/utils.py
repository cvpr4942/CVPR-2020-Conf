import os
import json
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype='float32')
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype='float32')
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path,
              gray=False,
              desired_size=None,
              expand=False,
              view=False):
    """
    Util function for loading RGB or Gray images into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    if gray:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if gray:
        x = np.expand_dims(x, axis=-1)
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x, gray=False):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB or Gray img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    if gray:
        return Image.fromarray(x.squeeze().astype('uint8'), 'L')
    return Image.fromarray(x.astype('uint8'), 'RGB')


def plot_omniglot_pairs(pair1, pair2, labels, name=None, save=False):
    num_rows = 2
    num_cols = pair1.shape[0]

    fig, big_axes = plt.subplots(
        figsize=(8.0, 8.0), nrows=num_rows, ncols=1, sharey=True,
    )

    for i, big_ax in enumerate(big_axes):
        xlabel = 'Same' if labels[i] == 1 else 'Different'
        big_ax.set_title(xlabel, fontsize=12)

        # turn off axis lines and ticks of the big subplot
        big_ax.tick_params(
            labelcolor=(1., 1., 1., 0.0), top='off',
            bottom='off', left='off', right='off'
        )
        big_ax._frameon = False

    for i in range(num_rows):
        # seperate the pair
        left, right = pair1[i], pair2[i]

        # create left subplot
        ax = fig.add_subplot(num_rows, num_cols, 2*i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(left, cmap="Greys_r")

        # create right subplot
        ax = fig.add_subplot(num_rows, num_cols, 2*(i+1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(right, cmap="Greys_r")

    fig.set_facecolor('w')
    plt.tight_layout()
    if save:
        plot_dir = './plots/'
        plt.savefig(plot_dir + name, format='png', dpi=150)
    plt.show()


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def prepare_dirs(config):
    num_model = get_num_model(config)
    for path in [config.ckpt_dir, config.logs_dir, config.plot_dir]:
        path = os.path.join(path, num_model)
        if not os.path.exists(path):
            os.makedirs(path)
        if config.flush:
            shutil.rmtree(path)
            if not os.path.exists(path):
                os.makedirs(path)


def save_config(config, hyperparams):
    num_model = get_num_model(config)
    model_dir = os.path.join(config.ckpt_dir, num_model)
    filename = 'params.json'
    param_path = os.path.join(model_dir, filename)

    if not os.path.isfile(param_path):
        print("[*] Model Checkpoint Dir: {}".format(model_dir))
        print("[*] Param Path: {}".format(param_path))

        all_params = config.__dict__
        all_params.update(hyperparams)
        with open(param_path, 'w') as fp:
            json.dump(all_params, fp, indent=4, sort_keys=True)
    else:
        raise ValueError


def load_config(config):
    num_model = get_num_model(config)
    model_dir = os.path.join(config.ckpt_dir, num_model)
    filename = 'params.json'
    param_path = os.path.join(model_dir, filename)
    params = json.load(open(param_path))
    print("[*] Loaded layer hyperparameters.")
    wanted_keys = [
        'layer_end_momentums', 'layer_init_lrs', 'layer_l2_regs'
    ]
    hyperparams = dict((k, params[k]) for k in wanted_keys if k in params)
    return hyperparams


def get_num_model(config):
    num_model = config.num_model
    error_msg = "[!] model number must be >= 1."
    assert num_model > 0, error_msg
    return 'exp_' + str(num_model)


# adapted from https://bit.ly/2pP5qki
class MacOSFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("Writing {} total bytes".format(n))
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("Writing bytes [{}, {}]".format(idx, idx+batch_size))
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size
        print("Done!")


# adapted from https://bit.ly/2pP5qki
def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


# adapted from https://bit.ly/2pP5qki
def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))
