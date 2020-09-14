import h5py
import numpy as np
import os
import nibabel as nib
from skimage import transform
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from networks import utils

def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=2):
    mask = np.asarray(mask)
    distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)
    weights = alpha + beta * np.exp(-(distance_to_border ** 2 / omega ** 2))
    return np.asarray(weights, dtype='float32')

if __name__ == '__main__':
    fold_num = 5
    train_num = 168
    val_num = 4
    test_num = 43
    total_num = 215
    mean = 466.0
    std = 379.0

    depth = 18
    height = 256
    width = 512
    class_num = 20
    mrDir = '/public/pangshumao/data/five-fold/high_resolution/in/nii/original_mr'
    maskDir = '/public/pangshumao/data/five-fold/high_resolution/in/nii/mask'
    outDir = '/public/pangshumao/data/five-fold/high_resolution/in/h5py'
    foldIndDir = '/public/pangshumao/data/five-fold/high_resolution'

    np.random.seed(0)

    ind = np.random.permutation(total_num)
    for i in range(fold_num):
        test_ind = ind[i * test_num : (i + 1) * test_num]
        test_ind = np.sort(test_ind)
        diff_ind = np.setdiff1d(ind, test_ind)

        val_ind = diff_ind[0:val_num]
        train_ind = np.setdiff1d(diff_ind, val_ind)

        train_ind = list(train_ind + 1)
        val_ind = list(val_ind + 1)
        test_ind = list(test_ind + 1)

        np.savez(os.path.join(foldIndDir, 'split_ind_fold' + str(i + 1) + '.npz'), train_ind=train_ind,
                                                                                val_ind=val_ind, test_ind=test_ind)

        try:
            f = h5py.File(os.path.join(outDir, 'data_fold' + str(i + 1) + '.h5'), 'w')
            g_train = f.create_group('train')
            g_val = f.create_group('val')
            g_test = f.create_group('test')

            # For training data
            flag = True
            for j in train_ind:
                print('processing fold %d, case %d' % (i + 1, j))
                mr = nib.load(os.path.join(mrDir, 'Case' + str(j) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
                mask = nib.load(os.path.join(maskDir, 'mask_case' + str(j) + '.nii.gz')).get_data().transpose(2, 0,
                                                                                                              1)  # [d, h, w]

                d, h, w = mr.shape
                shape = np.array([[d, h, w]])

                start_h = int(h / 4.)
                end_h = -int(h / 4.)
                mr = mr[:, start_h:end_h, :]  # [d, h/2, w]
                mask = mask[:, start_h:end_h, :]  # [d, h/2, w]

                # resize the data to the same shape
                mr_resize = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                             mode='constant').astype(np.float32)  # [d, height, width]

                mask_resize = transform.resize(mask.astype(np.float), (d, height, width), order=0, anti_aliasing=False,
                                               mode='constant').astype(np.float32)  # [d, height, width]


                # pad the data to has a same depth
                mr_out = np.zeros((1, 1, depth, height, width), dtype=np.float32)
                mask_out = np.zeros((1, depth, height, width), dtype=np.uint8)

                delata = depth - d
                start_d = int(math.ceil(delata / 2.))
                end_d = -int(delata - start_d)
                if end_d == 0:
                    end_d = None
                mr_out[0, 0, start_d: end_d, :, :] = mr_resize
                mr_out -= mean
                mr_out /= std
                mask_out[0, start_d: end_d, :, :] = mask_resize



                if flag:
                    flag = False
                    g_train.create_dataset('mr', data=mr_out,
                                           maxshape=(train_num, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3],
                                                     mr_out.shape[4]),
                                           chunks=(1, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3], mr_out.shape[4]))

                    g_train.create_dataset('mask', data=mask_out,
                                           maxshape=(train_num, mask_out.shape[1], mask_out.shape[2], mask_out.shape[3]),
                                           chunks=(1, mask_out.shape[1], mask_out.shape[2], mask_out.shape[3]))

                    g_train.create_dataset('shape', data=shape,
                                           maxshape=(train_num, shape.shape[1]),
                                           chunks=(1, shape.shape[1]))
                else:
                    g_train['mr'].resize(g_train['mr'].shape[0] + mr_out.shape[0], axis=0)
                    g_train['mr'][-mr_out.shape[0]:] = mr_out

                    g_train['mask'].resize(g_train['mask'].shape[0] + mask_out.shape[0], axis=0)
                    g_train['mask'][-mask_out.shape[0]:] = mask_out

                    g_train['shape'].resize(g_train['shape'].shape[0] + shape.shape[0], axis=0)
                    g_train['shape'][-shape.shape[0]:] = shape

            # For val data
            flag = True
            for j in val_ind:
                print('processing fold %d, case %d' % (i + 1, j))
                mr = nib.load(os.path.join(mrDir, 'Case' + str(j) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
                mask = nib.load(os.path.join(maskDir, 'mask_case' + str(j) + '.nii.gz')).get_data().transpose(2, 0,
                                                                                                              1)  # [d, h, w]

                d, h, w = mr.shape
                shape = np.array([[d, h, w]])

                start_h = int(h / 4.)
                end_h = -int(h / 4.)
                mr = mr[:, start_h:end_h, :]  # [d, h/2, w]
                mask = mask[:, start_h:end_h, :]  # [d, h/2, w]

                # resize the data to the same shape
                mr_resize = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                             mode='constant').astype(np.float32)  # [d, height, width]

                mask_resize = transform.resize(mask.astype(np.float), (d, height, width), order=0, anti_aliasing=False,
                                               mode='constant').astype(np.float32)  # [d, height, width]


                # pad the data to has a same depth
                mr_out = np.zeros((1, 1, depth, height, width), dtype=np.float32)
                mask_out = np.zeros((1, depth, height, width), dtype=np.uint8)

                delata = depth - d
                start_d = int(math.ceil(delata / 2.))
                end_d = -int(delata - start_d)
                if end_d == 0:
                    end_d = None
                mr_out[0, 0, start_d: end_d, :, :] = mr_resize
                mr_out -= mean
                mr_out /= std
                mask_out[0, start_d: end_d, :, :] = mask_resize


                if flag:
                    flag = False
                    g_val.create_dataset('mr', data=mr_out,
                                           maxshape=(val_num, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3],
                                                     mr_out.shape[4]),
                                           chunks=(1, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3], mr_out.shape[4]))

                    g_val.create_dataset('mask', data=mask_out,
                                           maxshape=(val_num, mask_out.shape[1], mask_out.shape[2], mask_out.shape[3]),
                                           chunks=(1, mask_out.shape[1], mask_out.shape[2], mask_out.shape[3]))

                    g_val.create_dataset('shape', data=shape,
                                           maxshape=(val_num, shape.shape[1]),
                                           chunks=(1, shape.shape[1]))
                else:
                    g_val['mr'].resize(g_val['mr'].shape[0] + mr_out.shape[0], axis=0)
                    g_val['mr'][-mr_out.shape[0]:] = mr_out

                    g_val['mask'].resize(g_val['mask'].shape[0] + mask_out.shape[0], axis=0)
                    g_val['mask'][-mask_out.shape[0]:] = mask_out

                    g_val['shape'].resize(g_val['shape'].shape[0] + shape.shape[0], axis=0)
                    g_val['shape'][-shape.shape[0]:] = shape

            # For test data
            flag = True
            for j in test_ind:
                print('processing fold %d, case %d' % (i + 1, j))
                mr = nib.load(os.path.join(mrDir, 'Case' + str(j) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
                mask = nib.load(os.path.join(maskDir, 'mask_case' + str(j) + '.nii.gz')).get_data().transpose(2, 0,
                                                                                                              1)  # [d, h, w]

                d, h, w = mr.shape
                shape = np.array([[d, h, w]])

                start_h = int(h / 4.)
                end_h = -int(h / 4.)
                mr = mr[:, start_h:end_h, :]  # [d, h/2, w]
                mask = mask[:, start_h:end_h, :]  # [d, h/2, w]

                # resize the data to the same shape
                mr_resize = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                             mode='constant').astype(np.float32)  # [d, height, width]

                mask_resize = transform.resize(mask.astype(np.float), (d, height, width), order=0, anti_aliasing=False,
                                               mode='constant').astype(np.float32)  # [d, height, width]


                # pad the data to has a same depth
                mr_out = np.zeros((1, 1, depth, height, width), dtype=np.float32)
                mask_out = np.zeros((1, depth, height, width), dtype=np.uint8)

                delata = depth - d
                start_d = int(math.ceil(delata / 2.))
                end_d = -int(delata - start_d)
                if end_d == 0:
                    end_d = None
                mr_out[0, 0, start_d: end_d, :, :] = mr_resize
                mr_out -= mean
                mr_out /= std
                mask_out[0, start_d: end_d, :, :] = mask_resize

                if flag:
                    flag = False
                    g_test.create_dataset('mr', data=mr_out,
                                           maxshape=(test_num, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3],
                                                     mr_out.shape[4]),
                                           chunks=(1, mr_out.shape[1], mr_out.shape[2], mr_out.shape[3], mr_out.shape[4]))

                    g_test.create_dataset('mask', data=mask_out,
                                           maxshape=(test_num, mask_out.shape[1], mask_out.shape[2], mask_out.shape[3]),
                                           chunks=(1, mask_out.shape[1], mask_out.shape[2], mask_out.shape[3]))

                    g_test.create_dataset('shape', data=shape,
                                           maxshape=(test_num, shape.shape[1]),
                                           chunks=(1, shape.shape[1]))
                else:
                    g_test['mr'].resize(g_test['mr'].shape[0] + mr_out.shape[0], axis=0)
                    g_test['mr'][-mr_out.shape[0]:] = mr_out

                    g_test['mask'].resize(g_test['mask'].shape[0] + mask_out.shape[0], axis=0)
                    g_test['mask'][-mask_out.shape[0]:] = mask_out

                    g_test['shape'].resize(g_test['shape'].shape[0] + shape.shape[0], axis=0)
                    g_test['shape'][-shape.shape[0]:] = shape
        finally:
            f.close()

