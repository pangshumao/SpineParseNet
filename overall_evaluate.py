import numpy as np
import os
from scipy.stats import ttest_rel as ttest
import nibabel as nib
from networks.utils import dices_each_class, assds_each_class
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/five-fold',
                        help="the data dir")

    # parser.add_argument("--coarse_identifier", type=str,
    #                     default='DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_'
    #                             'Adam_lr_0.001_pretrained',
    #                     help="coarse identifier")

    parser.add_argument("--coarse_identifier", type=str,
                        default='DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_0_loss_CrossEntropyLoss_Adam_lr_0.001_pretrained',
                        help="coarse identifier")

    parser.add_argument("--stage", type=str, default='fine',
                        help="fine or coarse")

    parser.add_argument("--model", type=str, default='ResidualUNet2D_Adam_lr_0.0001_weight_decay_0.0001_augment',
                        help="the model name")

    args = parser.parse_args()

    stage = args.stage
    model = args.model
    dataDir = args.data_dir

    maskDir = os.path.join(dataDir, stage, 'in/nii/mask')
    if args.coarse_identifier != 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                 'CrossEntropyLoss_Adam_lr_0.001_pretrained' and stage == 'fine':
        outDir = os.path.join(dataDir, stage, 'result_' + args.coarse_identifier)
    else:
        outDir = os.path.join(dataDir, stage, 'result')


    class_name = ['background', # 0
                  'S', # 1
                  'L5', # 2
                  'L4', # 3
                  'L3', # 4
                  'L2', # 5
                  'L1', # 6
                  'T12', # 7
                  'T11', # 8
                  'T10', # 9
                  'T9', # 10
                  'L5/S', # 11
                  'L4/L5', # 12
                  'L3/L4', # 13
                  'L2/L3', # 14
                  'L1/L2', # 15
                  'T12/L1', # 16
                  'T11/T12', # 17
                  'T10/T11', # 18
                  'T9/T10'] # 19

    dice_count = np.zeros((20), dtype=np.float32)
    assd_count = np.zeros((20), dtype=np.float32)
    all_dice = []
    all_assd = []
    dice_sum = np.zeros((20), dtype=np.float32)
    assd_sum = np.zeros((20), dtype=np.float32)
    mean_assds = []
    mean_dices = []
    for fold_ind in range(1, 6):
        foldIndData = np.load(os.path.join(dataDir, stage, 'split_ind_fold' + str(fold_ind) + '.npz'))
        train_ind = foldIndData['train_ind']
        val_ind = foldIndData['val_ind']
        test_ind = foldIndData['test_ind']

        for i in test_ind:
            print('processing fold%s, case%d' % (fold_ind, i))
            mask_nii = nib.load(os.sep.join([maskDir, 'mask_case' + str(i) + '.nii.gz']))
            mask = mask_nii.get_data()
            voxel_size = tuple(mask_nii.header['pixdim'][1:4])

            if args.coarse_identifier != 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                         'CrossEntropyLoss_Adam_lr_0.001_pretrained' and stage == 'fine':
                seg_nii = nib.load(os.sep.join(
                    [dataDir, stage, 'out', 'fold' + str(fold_ind) + '_' + args.coarse_identifier, model, 'seg_Case' + str(i) + '.nii.gz']))
            else:
                seg_nii = nib.load(os.sep.join([dataDir, stage, 'out', 'fold' + str(fold_ind), model, 'seg_Case' + str(i) + '.nii.gz']))
            seg = seg_nii.get_data()

            dice = dices_each_class(seg, mask, class_num=20, empty_value=-1.0)
            all_dice.append(dice)
            temp_count = dice > -1.0
            temp_count = temp_count.astype(np.float32)
            dice_count += temp_count
            temp_dice = np.where(dice == -1.0, 0.0, dice)
            dice_sum += temp_dice
            dice = np.where(dice == -1.0, np.nan, dice)
            mean_dice = np.nanmean(dice[1:])
            mean_dices.append(mean_dice)

            assd = assds_each_class(seg, mask, class_num=20, voxel_size=voxel_size, empty_value=-1.0, connectivity=2)
            all_assd.append(assd)

            temp_count = assd > -1.0
            temp_count = temp_count.astype(np.float32)
            assd_count += temp_count

            temp_assd = np.where(assd == -1.0, 0.0, assd)
            assd_sum += temp_assd
            assd = np.where(assd == -1.0, np.nan, assd)
            mean_assd = np.nanmean(assd[1:])
            mean_assds.append(mean_assd)
    all_class_dice = dice_sum / dice_count

    all_class_assd = assd_sum / assd_count

    for i in range(0, 20):
        print('%s dice = %.4f' % (class_name[i], all_class_dice[i]))
    print('image-level mean dice = %.4f' % np.mean(mean_dices))
    print('...........................................................')

    for i in range(0, 20):
        print('%s assd = %.4f' % (class_name[i], all_class_assd[i]))
    print('image-level mean assd = %.4f' % np.mean(mean_assds))

    if not os.path.exists(os.path.join(outDir, model)):
        os.makedirs(os.path.join(outDir, model))

    np.savez(os.path.join(outDir, model, 'dices_assds.npz'), all_class_dice=all_class_dice, mean_dice=np.mean(mean_dices),
             all_class_assd=all_class_assd, mean_assd=np.mean(mean_assds), all_dice=all_dice, all_assd=all_assd)





