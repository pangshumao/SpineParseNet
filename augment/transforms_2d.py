from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
import importlib

import numpy as np
import torch
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
import cv2
from torchvision.transforms import Compose
# from datasets.h5 import load_data
import matplotlib.pyplot as plt


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=10, interpolation='cubic', **kwargs):

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        if interpolation == 'cubic':
            self.interpolation = cv2.INTER_CUBIC
        elif interpolation == 'nearest':
            self.interpolation = cv2.INTER_NEAREST

    def __call__(self, m):
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 2:
            m = self.rotate(m, angle)
        else:
            channels = [self.rotate(m[c], angle) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

    def rotate(self, image, angle, center=None, scale=1.0):
        '''
        @author Shumao Pang, Southern Medical University, pangshumao@126.com
        :param image: a numpy array with shape of h * w
        :param angle:
        :param center:
        :param scale:
        :return:
        '''
        h, w = image.shape
        if center is None:
            center = (int(w / 2), int(h / 2))
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image.astype(np.float32), M, (w, h), flags=self.interpolation, borderMode=cv2.BORDER_REFLECT)
        return rotated.astype(image.dtype)


class RandomContrast:
    """
        Adjust the brightness of an image by a random factor.
    """

    def __init__(self, random_state, factor=0.5, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.factor = factor
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            brightness_factor = self.factor + self.random_state.uniform()
            return np.clip(m * brightness_factor, -10, 10)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=3 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=15, sigma=3, execution_probability=0.3, **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [2, 3]


            if m.ndim == 2:
                dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant",
                                     cval=0) * self.alpha
                dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant",
                                     cval=0) * self.alpha
                y_dim, x_dim = m.shape
                y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
                indices = y + dy, x + dx
                m = map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                dy = gaussian_filter(self.random_state.randn(*m.shape[1:]), self.sigma, mode="constant",
                                     cval=0) * self.alpha
                dx = gaussian_filter(self.random_state.randn(*m.shape[1:]), self.sigma, mode="constant",
                                     cval=0) * self.alpha
                y_dim, x_dim = m.shape[1:]
                y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
                indices = y + dy, x + dx
                channels = [map_coordinates(m[c], indices, order=self.spline_order, mode='reflect') for c in range(m.shape[0])]
                m = np.stack(channels, axis=0)
        return m



class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-4, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / (self.std + self.eps)


class GaussianNoise:
    def __init__(self, random_state, max_sigma, max_value=255, **kwargs):
        self.random_state = random_state
        self.max_sigma = max_sigma
        self.max_value = max_value

    def __call__(self, m):
        # pick std dev from [0; max_sigma]
        std = self.random_state.randint(self.max_sigma)
        gaussian_noise = self.random_state.normal(0, std, m.shape)
        noisy_m = m + gaussian_noise
        return np.clip(noisy_m, 0, self.max_value).astype(m.dtype)



class Identity:
    def __call__(self, m):
        return m


def get_transformer(config, mean, std, phase):
    if phase == 'val':
        phase = 'test'

    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    return Transformer(phase_config, mean, std)


class Transformer:
    def __init__(self, phase_config, mean, std):
        self.phase_config = phase_config
        self.config_base = {'mean': mean, 'std': std}
        self.seed = 47

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def seg_transform(self):
        return self._create_transform('seg')

    def weight_transform(self):
        return self._create_transform('weight')

    def unary_transform(self):
        return self._create_transform('unary')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('augment.transforms_2d')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input


if __name__ == '__main__':
    pass

    # data_path = '/public/pangshumao/data/five-fold/fine/in/h5py/fold1_data.h5'
    # randomState = np.random.RandomState(47)
    # randomRotate = RandomRotate(randomState, angle_spectrum=15, interpolation='cubic')
    # randomContrast = RandomContrast(randomState, execution_probability=1.0)
    # elasticDeformation = ElasticDeformation(randomState, spline_order=3, execution_probability=1.0)
    # try:
    #     train_data_loader, val_data_loader, test_data_loader, f = load_data(data_path, batch_size=1, shuffle=False)
    #     for i, t in enumerate(test_data_loader):
    #         mr, feature, unary, mask = t
    #         mr = mr.numpy().squeeze()
    #         feature = feature.numpy().squeeze()
    #         unary = unary.numpy().squeeze()
    #         mask = mask.numpy().squeeze()
    #
    #         mr_rot = randomRotate(mr)
    #         plt.subplot(121)
    #         plt.imshow(mr, cmap='gray')
    #         plt.subplot(122)
    #         plt.imshow(mr_rot, cmap='gray')
    #         plt.show()
    #         pass
    # finally:
    #     f.colse()

