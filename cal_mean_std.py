from matplotlib import pyplot as plt
import os
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    dataDir = '/public/pangshumao/data/spine'
    images = []
    for i in range(1,216):
        print('loading case%d......................................'%i)
        nii = nib.load(os.path.join(dataDir, 'Case' + str(i) + '.nii.gz'))
        image = nii.get_data().transpose((2, 0, 1))
        d, h, w = image.shape
        image = image[:, int(h / 4.): -int(h / 4.), :]
        # plt.imshow(image[6])
        # plt.show()
        images.extend(image.flatten())
    print('mean = %.4f' % np.mean(images))
    print('std = %.4f' % np.std(images))

