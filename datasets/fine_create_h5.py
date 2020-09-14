import sys

sys.path.append('networks')
import h5py
import numpy as np
import os
import nibabel as nib
from skimage import transform
import matplotlib.pyplot as plt
from scipy import ndimage
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def np_onehot(label, num_classes):
    return np.eye(num_classes)[label.astype(np.int32)]

def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=2):
    mask = np.asarray(mask)
    distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)
    weights = alpha + beta * np.exp(-(distance_to_border ** 2 / omega ** 2))
    return np.asarray(weights, dtype='float32')

def compute_weight(mask, classes_num=20, alpha=1, beta=40, omega=6):
    '''

    :param mask: a numpy array with shape of [d, h, w]
    :param classes_num:
    :param alpha:
    :param beta:
    :param omega:
    :return: a numpy array with shape of [d, h, w]
    '''
    assert mask.ndim == 3
    d, h, w = mask.shape
    mask_one_hot = np_onehot(mask, classes_num)  # [d, height, width, 20]
    mask_one_hot = np.transpose(mask_one_hot, axes=(0, 3, 1, 2))  # [d, 20, height, width]
    weights = np.zeros((d, h, w), dtype=np.float32)
    for i in range(d):
        weight = 0
        count = 0
        for j in range(20):
            if j not in mask:
                continue

            weight += compute_distance_weight_matrix(mask_one_hot[i, j, :, :], alpha=alpha, beta=beta, omega=omega)
            count += 1
        weight /= count
        weights[i] = weight
    return weights

def test_weight():
    height = 256
    width = 512

    fineDir = '/public/pangshumao/data/five-fold/fine'
    maskDir = os.path.join(fineDir, 'in/nii/mask')
    mask = nib.load(os.path.join(maskDir, 'mask_case' + str(1) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
    d, h, w = mask.shape
    start_h = int(h / 4.)
    end_h = -int(h / 4.)
    mask = mask[:, start_h:end_h, :]  # [d, h/2, w]
    mask = transform.resize(mask.astype(np.float), (d, height, width), order=0, anti_aliasing=False,
                            mode='constant').astype(np.float32)  # [d, height, width]
    weight = compute_weight(mask)
    for i in range(d):
        plt.subplot(121)
        plt.imshow(mask[i])
        plt.subplot(122)
        plt.imshow(weight[i])
        plt.show()


def test_read_h5(fold_ind):
    fileDir = '/public/pangshumao/data/five-fold/fine/in/h5py'
    f = h5py.File(os.path.join(fileDir, 'fold' + str(fold_ind) + '_data.h5'), 'r')
    train_mr = f['train']['mr']
    val_mr = f['val']['mr']
    test_mr = f['test']['mr']

    train_num = train_mr.shape[0]
    val_num = val_mr.shape[0]
    test_num = test_mr.shape[0]
    np.savez(os.path.join(fileDir, 'fold' + str(fold_ind) + '_sample_num.npz'), train_num=train_num, val_num=val_num, test_num=test_num)

    # train_mr = f['train']['mr']
    # train_unary = f['train']['unary']
    # train_mask = f['train']['mask']
    # train_feature = f['train']['feature']
    # print(train_mr.shape)
    # print(train_unary.shape)
    # print(train_mask.shape)
    # print(train_feature.shape)


if __name__ == '__main__':
    # test_weight()
    # for i in range(1, 6):
    #     test_read_h5(i)

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--coarse_identifier", type=str,
                        default='DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_Adam_lr_0.001_pretrained',
                        help="coarse identifier")

    parser.add_argument("--coarse_dir", type=str, default='/public/pangshumao/data/five-fold/coarse',
                        help="coarse dir")

    parser.add_argument("--fine_dir", type=str, default='/public/pangshumao/data/five-fold/fine',
                        help="fine dir")


    args = parser.parse_args()

    coarseDir = args.coarse_dir
    fineDir = args.fine_dir

    mrDir = os.path.join(fineDir, 'in/nii/original_mr')
    maskDir = os.path.join(fineDir, 'in/nii/mask')
    identifier = args.coarse_identifier

    height = 256
    width = 512

    mean = 466.0
    std = 379.0

    for fold_ind in range(1, 6):
        foldIndData = np.load(os.path.join(fineDir, 'split_ind_fold' + str(fold_ind) + '.npz'))
        unaryDir = os.path.join(coarseDir, 'out', 'fold' + str(fold_ind), identifier)
        outDir = os.path.join(fineDir, 'in', 'h5py')

        train_ind = foldIndData['train_ind']
        val_ind = foldIndData['val_ind']
        test_ind = foldIndData['test_ind']

        # calculate the slices number
        train_slice_num = 0
        val_slice_num = 0
        test_slice_num = 0
        for i in train_ind:
            temp = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
            train_slice_num += temp.shape[0]

        for i in val_ind:
            temp = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
            val_slice_num += temp.shape[0]

        for i in test_ind:
            temp = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
            test_slice_num += temp.shape[0]

        try:
            if identifier == 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_Adam_lr_0.001_pretrained':
                f = h5py.File(os.path.join(outDir, 'fold' + str(fold_ind) + '_data.h5'), 'w')
            else:
                f = h5py.File(os.path.join(outDir, 'fold' + str(fold_ind) + '_data_' + identifier + '.h5'), 'w')
            g_train = f.create_group('train')

            g_val = f.create_group('val')

            g_test = f.create_group('test')

            # For training data
            flag = True
            for i in train_ind:
                print('processing training fold%d, case%d' % (fold_ind, i))
                mr = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]
                mr -= mean
                mr /= std
                mask = nib.load(os.path.join(maskDir, 'mask_case' + str(i) + '.nii.gz')).get_data().transpose(2, 0, 1)  # [d, h, w]

                unary = np.load(os.path.join(unaryDir, 'Case' + str(i) + '_logit.npz'))['logit'] # [1, 20, d, 128, 256]

                d, h, w = mr.shape
                start_h = int(h / 4.)
                end_h = -int(h / 4.)
                mr = mr[:, start_h:end_h, :] # [d, h/2, w]
                mask = mask[:, start_h:end_h, :] # [d, h/2, w]

                # resize the data to the same shape
                mr = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                       mode='constant') # [d, height, width]
                mr = np.reshape(mr, (d, 1, height, width)) # [d, 1, height, width]

                unary = np.squeeze(unary)

                mask = transform.resize(mask.astype(np.float), (d, height, width), order=0, anti_aliasing=False,
                                         mode='constant').astype(np.float32) # [d, height, width]
                weight = compute_weight(mask) # [d, height, width]

                # transpose the data
                unary = unary.transpose((1, 0, 2, 3)) # [d, 20, height, width]

                if flag:
                    flag = False
                    g_train.create_dataset('mr', data=mr, maxshape=(train_slice_num, mr.shape[1], mr.shape[2], mr.shape[3]),
                                           chunks=(1, mr.shape[1], mr.shape[2], mr.shape[3]))
                    g_train.create_dataset('weight', data=weight, maxshape=(train_slice_num, weight.shape[1], weight.shape[2]),
                                           chunks=(1, weight.shape[1], weight.shape[2]))
                    g_train.create_dataset('unary', data=unary, maxshape=(train_slice_num, unary.shape[1], unary.shape[2], unary.shape[3]),
                                           chunks=(1, unary.shape[1], unary.shape[2], unary.shape[3]))
                    g_train.create_dataset('mask', data=mask, maxshape=(train_slice_num, mask.shape[1], mask.shape[2]),
                                           chunks=(1, mask.shape[1], mask.shape[2]))
                else:
                    g_train['mr'].resize(g_train['mr'].shape[0] + mr.shape[0], axis=0)
                    g_train['mr'][-mr.shape[0] : ] = mr

                    g_train['weight'].resize(g_train['weight'].shape[0] + weight.shape[0], axis=0)
                    g_train['weight'][-weight.shape[0]:] = weight

                    g_train['unary'].resize(g_train['unary'].shape[0] + unary.shape[0], axis=0)
                    g_train['unary'][-unary.shape[0]:] = unary

                    g_train['mask'].resize(g_train['mask'].shape[0] + mask.shape[0], axis=0)
                    g_train['mask'][-mask.shape[0]:] = mask

            # For val data
            flag = True
            for i in val_ind:
                print('processing val fold%d, case%d' % (fold_ind, i))
                mr = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz')).get_data().transpose(2, 0,
                                                                                                     1)  # [d, h, w]
                mr -= mean
                mr /= std
                mask = nib.load(os.path.join(maskDir, 'mask_case' + str(i) + '.nii.gz')).get_data().transpose(2, 0,
                                                                                                              1)  # [d, h, w]

                unary = np.load(os.path.join(unaryDir, 'Case' + str(i) + '_logit.npz'))['logit']  # [1, 20, d, 128, 256]

                d, h, w = mr.shape
                start_h = int(h / 4.)
                end_h = -int(h / 4.)
                mr = mr[:, start_h:end_h, :]  # [d, h/2, w]
                mask = mask[:, start_h:end_h, :]  # [d, h/2, w]

                # resize the data to the same shape
                mr = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                      mode='constant')  # [d, height, width]
                mr = np.reshape(mr, (d, 1, height, width))  # [d, 1, height, width]

                unary = np.squeeze(unary)

                mask = transform.resize(mask.astype(np.float), (d, height, width), order=0, anti_aliasing=False,
                                        mode='constant').astype(np.float32)  # [d, height, width]
                weight = compute_weight(mask)  # [d, height, width]

                # transpose the data
                unary = unary.transpose((1, 0, 2, 3))  # [d, 20, height, width]

                if flag:
                    flag = False
                    g_val.create_dataset('mr', data=mr,
                                           maxshape=(val_slice_num, mr.shape[1], mr.shape[2], mr.shape[3]),
                                           chunks=(1, mr.shape[1], mr.shape[2], mr.shape[3]))
                    g_val.create_dataset('weight', data=weight,
                                         maxshape=(val_slice_num, weight.shape[1], weight.shape[2]),
                                           chunks=(1, weight.shape[1], weight.shape[2]))
                    g_val.create_dataset('unary', data=unary,
                                           maxshape=(val_slice_num, unary.shape[1], unary.shape[2], unary.shape[3]),
                                           chunks=(1, unary.shape[1], unary.shape[2], unary.shape[3]))
                    g_val.create_dataset('mask', data=mask, maxshape=(val_slice_num, mask.shape[1], mask.shape[2]),
                                           chunks=(1, mask.shape[1], mask.shape[2]))
                else:
                    g_val['mr'].resize(g_val['mr'].shape[0] + mr.shape[0], axis=0)
                    g_val['mr'][-mr.shape[0]:] = mr

                    g_val['weight'].resize(g_val['weight'].shape[0] + weight.shape[0], axis=0)
                    g_val['weight'][-weight.shape[0]:] = weight

                    g_val['unary'].resize(g_val['unary'].shape[0] + unary.shape[0], axis=0)
                    g_val['unary'][-unary.shape[0]:] = unary

                    g_val['mask'].resize(g_val['mask'].shape[0] + mask.shape[0], axis=0)
                    g_val['mask'][-mask.shape[0]:] = mask

            # For test data
            flag = True
            for i in test_ind:
                print('processing test fold%d, case%d' % (fold_ind, i))
                mr = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz')).get_data().transpose(2, 0,
                                                                                                     1)  # [d, h, w]
                mr -= mean
                mr /= std
                mask = nib.load(os.path.join(maskDir, 'mask_case' + str(i) + '.nii.gz')).get_data().transpose(2, 0,
                                                                                                              1)  # [d, h, w]

                unary = np.load(os.path.join(unaryDir, 'Case' + str(i) + '_logit.npz'))['logit']  # [1, 20, d, 128, 256]

                d, h, w = mr.shape
                start_h = int(h / 4.)
                end_h = -int(h / 4.)
                mr = mr[:, start_h:end_h, :]  # [d, h/2, w]
                mask = mask[:, start_h:end_h, :]  # [d, h/2, w]

                # resize the data to the same shape
                mr = transform.resize(mr.astype(np.float), (d, height, width), order=3,
                                      mode='constant')  # [d, height, width]
                mr = np.reshape(mr, (d, 1, height, width))  # [d, 1, height, width]

                unary = np.squeeze(unary)

                mask = transform.resize(mask.astype(np.float), (d, height, width), order=0, anti_aliasing=False,
                                        mode='constant').astype(np.float32)  # [d, height, width]
                weight = compute_weight(mask)

                # transpose the data
                unary = unary.transpose((1, 0, 2, 3))  # [d, 20, height, width]

                if flag:
                    flag = False
                    g_test.create_dataset('mr', data=mr,
                                         maxshape=(test_slice_num, mr.shape[1], mr.shape[2], mr.shape[3]),
                                         chunks=(1, mr.shape[1], mr.shape[2], mr.shape[3]))
                    g_test.create_dataset('weight', data=weight,
                                         maxshape=(test_slice_num, weight.shape[1], weight.shape[2]),
                                         chunks=(1, weight.shape[1], weight.shape[2]))
                    g_test.create_dataset('unary', data=unary,
                                         maxshape=(test_slice_num, unary.shape[1], unary.shape[2], unary.shape[3]),
                                         chunks=(1, unary.shape[1], unary.shape[2], unary.shape[3]))
                    g_test.create_dataset('mask', data=mask, maxshape=(test_slice_num, mask.shape[1], mask.shape[2]),
                                         chunks=(1, mask.shape[1], mask.shape[2]))
                else:
                    g_test['mr'].resize(g_test['mr'].shape[0] + mr.shape[0], axis=0)
                    g_test['mr'][-mr.shape[0]:] = mr

                    g_test['weight'].resize(g_test['weight'].shape[0] + weight.shape[0], axis=0)
                    g_test['weight'][-weight.shape[0]:] = weight

                    g_test['unary'].resize(g_test['unary'].shape[0] + unary.shape[0], axis=0)
                    g_test['unary'][-unary.shape[0]:] = unary

                    g_test['mask'].resize(g_test['mask'].shape[0] + mask.shape[0], axis=0)
                    g_test['mask'][-mask.shape[0]:] = mask
        finally:
            f.close()
        print('Job done!!!')






