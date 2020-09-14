import numpy as np
import os
from scipy.stats import ttest_rel as ttest
import nibabel as nib
from networks.utils import evaluation_metrics_each_class
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
                        default='DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_Adam_lr_0.001_pretrained',
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

    ind_list = [0, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 19, 18, 17, 16, 15, 14, 13, 12, 11]

    dice_count = np.zeros((20), dtype=np.float32)
    assd_count = np.zeros((20), dtype=np.float32)
    all_dice = []
    all_precision = []
    all_recall = []

    subject_dices = []
    subject_precisions = []
    subject_recalls = []

    subject_ver_dices = []
    subject_ver_precisions = []
    subject_ver_recalls = []

    subject_ivd_dices = []
    subject_ivd_precisions = []
    subject_ivd_recalls = []
    for fold_ind in range(1, 6):
        foldIndData = np.load(os.path.join(dataDir, stage, 'split_ind_fold' + str(fold_ind) + '.npz'))
        train_ind = foldIndData['train_ind']
        val_ind = foldIndData['val_ind']
        test_ind = foldIndData['test_ind']

        for i in test_ind:
            print('processing fold%s, case%d' % (fold_ind, i))
            mask_nii = nib.load(os.sep.join([maskDir, 'mask_case' + str(i) + '.nii.gz']))
            mask = mask_nii.get_data()


            if args.coarse_identifier != 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                         'CrossEntropyLoss_Adam_lr_0.001_pretrained' and stage == 'fine':
                seg_nii = nib.load(os.sep.join(
                    [dataDir, stage, 'out', 'fold' + str(fold_ind) + '_' + args.coarse_identifier, model, 'seg_Case' + str(i) + '.nii.gz']))
            else:
                seg_nii = nib.load(os.sep.join([dataDir, stage, 'out', 'fold' + str(fold_ind), model, 'seg_Case' + str(i) + '.nii.gz']))
            seg = seg_nii.get_data()

            dice, precision, recall = evaluation_metrics_each_class(seg, mask, class_num=20, empty_value=-1.0)
            all_dice.append(dice)
            all_precision.append(precision)
            all_recall.append(recall)

            dice = np.where(dice == -1.0, np.nan, dice)
            print(np.nanmean(dice[1:]))
            precision = np.where(precision == -1.0, np.nan, precision)
            recall = np.where(recall == -1.0, np.nan, recall)

            subject_dices.append(np.nanmean(dice[1:]))
            subject_precisions.append(np.nanmean(precision[1:]))
            subject_recalls.append(np.nanmean(recall[1:]))

            subject_ver_dices.append(np.nanmean(dice[1:11]))
            subject_ver_precisions.append(np.nanmean(precision[1:11]))
            subject_ver_recalls.append(np.nanmean(recall[1:11]))

            subject_ivd_dices.append(np.nanmean(dice[11:]))
            subject_ivd_precisions.append(np.nanmean(precision[11:]))
            subject_ivd_recalls.append(np.nanmean(recall[11:]))

    class_dices_mean = np.nanmean([np.where(temp_dice == -1.0, np.nan, temp_dice) for temp_dice in all_dice], axis=0)
    class_precisions_mean = np.nanmean([np.where(temp_precision == -1.0, np.nan, temp_precision) for temp_precision in all_precision], axis=0)
    class_recalls_mean = np.nanmean([np.where(temp_recall == -1.0, np.nan, temp_recall) for temp_recall in all_recall], axis=0)

    class_dices_std = np.nanstd([np.where(temp_dice == -1.0, np.nan, temp_dice) for temp_dice in all_dice], axis=0)
    class_precisions_std = np.nanstd([np.where(temp_precision == -1.0, np.nan, temp_precision) for temp_precision in all_precision], axis=0)
    class_recalls_std = np.nanstd([np.where(temp_recall == -1.0, np.nan, temp_recall) for temp_recall in all_recall], axis=0)

    for i in range(0, 20):
        print('%s mean dice = %.2f +- %.2f' % (
        class_name[ind_list[i]], class_dices_mean[ind_list[i]] * 100, class_dices_std[ind_list[i]] * 100))

    print(
        'image-level ver mean dice = %.2f +- %.2f' % (np.mean(subject_ver_dices) * 100, np.std(subject_ver_dices) * 100))
    print(
        'image-level ivd mean dice = %.2f +- %.2f' % (np.mean(subject_ivd_dices) * 100, np.std(subject_ivd_dices) * 100))
    print('image-level total mean dice = %.2f +- %.2f' % (np.mean(subject_dices) * 100, np.std(subject_dices) * 100))
    print('...........................................................')


    for i in range(0, 20):
        print('%s mean precision = %.2f +- %.2f' % (
        class_name[ind_list[i]], class_precisions_mean[ind_list[i]] * 100, class_precisions_std[ind_list[i]] * 100))

    print(
        'image-level ver mean precision = %.2f +- %.2f' % (np.mean(subject_ver_precisions) * 100, np.std(subject_ver_precisions) * 100))
    print(
        'image-level ivd mean precision = %.2f +- %.2f' % (np.mean(subject_ivd_precisions) * 100, np.std(subject_ivd_precisions) * 100))
    print('image-level total mean precision = %.2f +- %.2f' % (np.mean(subject_precisions) * 100, np.std(subject_precisions) * 100))
    print('...........................................................')

    for i in range(0, 20):
        print('%s mean recall = %.2f +- %.2f' % (
        class_name[ind_list[i]], class_recalls_mean[ind_list[i]] * 100, class_recalls_std[ind_list[i]] * 100))

    print(
        'image-level ver mean recall = %.2f +- %.2f' % (np.mean(subject_ver_recalls) * 100, np.std(subject_ver_recalls) * 100))
    print(
        'image-level ivd mean recall = %.2f +- %.2f' % (np.mean(subject_ivd_recalls) * 100, np.std(subject_ivd_recalls) * 100))
    print('image-level total mean recall = %.2f +- %.2f' % (np.mean(subject_recalls) * 100, np.std(subject_recalls) * 100))

    if not os.path.exists(os.path.join(outDir, model)):
        os.makedirs(os.path.join(outDir, model))

    np.savez(os.path.join(outDir, model, 'all_evaluate_metrics.npz'), all_dice=all_dice,all_precision=all_precision,
             all_recall=all_recall,
             subject_dices=subject_dices, subject_precisions=subject_precisions, subject_recalls=subject_recalls,
             subject_ver_dices=subject_ver_dices, subject_ver_precisions=subject_ver_precisions, subject_ver_recalls=subject_ver_recalls,
             subject_ivd_dices=subject_ivd_dices, subject_ivd_precisions=subject_ivd_precisions, subject_ivd_recalls=subject_ivd_recalls,
             class_dices_mean=class_dices_mean, class_precisions_mean=class_precisions_mean, class_recalls_mean=class_recalls_mean,
             class_dices_std=class_dices_std, class_precisions_std=class_precisions_std, class_recalls_std=class_recalls_std)





