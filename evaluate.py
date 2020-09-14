import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from skimage import transform

# segDir = '/public/pangshumao/data/spine_out_deepLabv3_plus_gcn_3d/'
segDir = '/public/pangshumao/data/five-fold/fine/out/fold1/ResidualUNet2D_Adam_lr_0.0001_weight_decay_0.0001_augment'
maskDir = '/public/pangshumao/data/five-fold/fine/in/nii/mask'
indDir = '/public/pangshumao/data/five-fold/fine'

def get_dice(prediction, target, eps=1e-7):
    prediction = prediction.astype(np.float)
    target = target.astype(np.float)
    intersect = np.sum(prediction * target)
    return (2. * intersect / (np.sum(prediction) + np.sum(target) + eps))

if __name__ == '__main__':
    ver_dices = []
    disc_dices = []
    foldIndData = np.load(os.path.join(indDir, 'split_ind_fold' + str(1) + '.npz'))
    train_ind = foldIndData['train_ind']
    val_ind = foldIndData['val_ind']
    test_ind = foldIndData['test_ind']
    for i in test_ind:
        mask_nii = nib.load(os.sep.join([maskDir, 'mask_case' + str(i) + '.nii.gz']))
        mask = mask_nii.get_data()
        mask_remove_disc = np.where(mask > 10, 0, mask)
        mask_vertebrae = np.where(mask_remove_disc > 0, 1, 0)
        mask_disc = np.where(mask > 10, 1, 0)

        seg_nii = nib.load(os.sep.join([segDir, 'seg_Case' + str(i) + '.nii.gz']))
        seg = seg_nii.get_data()
        seg_remove_disc = np.where(seg > 10, 0, seg)
        seg_vertebrae = np.where(seg_remove_disc > 0, 1, 0)
        seg_disc = np.where(seg > 10, 1, 0)

        # h, w, d = seg.shape
        # mask_vertebrae = transform.resize(mask_vertebrae.astype(np.float), (256, 256, d), order=0, anti_aliasing=False)
        # mask_disc = transform.resize(mask_disc.astype(np.float), (256, 256, d), order=0, anti_aliasing=False)
        # seg_vertebrae = transform.resize(seg_vertebrae.astype(np.float), (256, 256, d), order=0, anti_aliasing=False)
        # seg_disc = transform.resize(seg_disc.astype(np.float), (256, 256, d), order=0, anti_aliasing=False)

        ver_dice = get_dice(seg_vertebrae, mask_vertebrae)
        disc_dice = get_dice(seg_disc, mask_disc)
        ver_dices.append(ver_dice)
        disc_dices.append(disc_dice)
        print('Case %d, vertebrae dice = %f, disc_dice = %f' % (i, ver_dice, disc_dice))
    print('.' * 100)
    print('mean vertebrae dice = %f' % np.mean(ver_dices))
    print('mean disc dice = %f' % np.mean(disc_dices))



