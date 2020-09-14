import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import augment.transforms as transforms
import matplotlib.pylab as plt

class H5Dataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """
    def __init__(self, data, return_weight, transformer_config, phase='train'):
        if return_weight:
            assert data['mr'].shape[0] == data['mask'].shape[0] == data['weight'].shape[0]
            self.weight = data['weight']
        else:
            assert data['mr'].shape[0] == data['mask'].shape[0]
        self.raw = data['mr']
        self.labels = data['mask']

        self.phase = phase
        self.return_weight = return_weight
        if self.phase == 'train':
            self.transformer = transforms.get_transformer(transformer_config, mean=0, std=1, phase=phase)
            self.raw_transform = self.transformer.raw_transform()
            self.label_transform = self.transformer.label_transform()
            if return_weight:
                self.weight_transform = self.transformer.weight_transform()
    def __getitem__(self, index):
        if self.phase == 'train':
            raw = np.squeeze(self.raw[index])
            raw_transformed = self._transform_image(raw, self.raw_transform)
            raw_transformed = np.expand_dims(raw_transformed, axis=0)
            label_transformed = self._transform_image(self.labels[index], self.label_transform)

            if self.return_weight:
                weight_transformed = self._transform_image(self.weight[index], self.weight_transform)
                # plt.subplot(131)
                # plt.imshow(raw_transformed[0, 7, :, :])
                # plt.subplot(132)
                # plt.imshow(label_transformed[7])
                # plt.subplot(133)
                # plt.imshow(weight_transformed[7])
                # plt.show()
                return raw_transformed, label_transformed.astype(np.long), weight_transformed
            else:
                return raw_transformed, label_transformed.astype(np.long)
        else:
            if self.return_weight:
                return self.raw[index], self.labels[index].astype(np.long), self.weight[index]
            else:
                return self.raw[index], self.labels[index].astype(np.long)

    def __len__(self):
        return self.raw.shape[0]

    @staticmethod
    def _transform_image(dataset, transformer):
        return transformer(dataset)

def load_data(filePath, transformer_config, return_weight=True, batch_size=1, num_workers=1, shuffle=True):
    f = h5py.File(filePath, 'r')
    train_data = f['train']
    val_data = f['val']
    test_data = f['test']
    train_set = H5Dataset(train_data, return_weight, transformer_config, phase='train')
    val_set = H5Dataset(val_data, return_weight, transformer_config, phase='val')
    test_set = H5Dataset(test_data, return_weight, transformer_config, phase='test')
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                      shuffle=shuffle)

    val_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                      shuffle=shuffle)

    testing_data_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                     shuffle=shuffle)
    return training_data_loader, val_data_loader, testing_data_loader, f

if __name__ == '__main__':
    inDir = '/public/pangshumao/data/five-fold/coarse/in/h5py'
    filePath = os.path.join(inDir, 'data_fold2.h5')

    try:
        train_data_loader, val_data_loader, test_data_loader, f = load_data(filePath)
        for i, t in enumerate(train_data_loader):
            print(i)
            mr,  mask, weight = t
            pass
    finally:
        f.close()

