import numpy as np
import os
from scipy.stats import ttest_rel as ttest
import nibabel as nib
from networks.utils import evaluation_accuracy
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


    subject_accuracies = []


    for fold_ind in range(1, 6):
        foldIndData = np.load(os.path.join(dataDir, stage, 'split_ind_fold' + str(fold_ind) + '.npz'))
        train_ind = foldIndData['train_ind']
        val_ind = foldIndData['val_ind']
        test_ind = foldIndData['test_ind']

        for i in test_ind:
            mask_nii = nib.load(os.sep.join([maskDir, 'mask_case' + str(i) + '.nii.gz']))
            mask = mask_nii.get_data()


            if args.coarse_identifier != 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                         'CrossEntropyLoss_Adam_lr_0.001_pretrained' and stage == 'fine':
                seg_nii = nib.load(os.sep.join(
                    [dataDir, stage, 'out', 'fold' + str(fold_ind) + '_' + args.coarse_identifier, model, 'seg_Case' + str(i) + '.nii.gz']))
            else:
                seg_nii = nib.load(os.sep.join([dataDir, stage, 'out', 'fold' + str(fold_ind), model, 'seg_Case' + str(i) + '.nii.gz']))
            seg = seg_nii.get_data()

            accuracy = evaluation_accuracy(seg, mask, class_num=20)

            print('fold%d, case%d, accuracy = %.4f' % (fold_ind, i, accuracy))

            subject_accuracies.append(accuracy)


    print('image-level total mean accuracy = %.2f +- %.2f' % (np.mean(subject_accuracies) * 100, np.std(subject_accuracies) * 100))
    print('...........................................................')



    if not os.path.exists(os.path.join(outDir, model)):
        os.makedirs(os.path.join(outDir, model))

    np.savez(os.path.join(outDir, model, 'all_evaluate_accuracies.npz'),
             subject_accuracies=subject_accuracies)





