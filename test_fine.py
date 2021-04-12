import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets.fine_h5 import load_data
import os
from networks.utils import get_logger, get_number_of_learnable_parameters
import torch
from networks import utils
import nibabel as nib
import torch.nn.functional as F
from skimage import transform
import math
from networks.unet_2d import UNet2D, ResidualUNet2D
from networks.deeplab_xception_skipconnection_2d import DeepLabv3_plus_skipconnection_2d
from networks.deeplab_xception_gcn_skipconnection_2d import DeepLabv3_plus_gcn_skipconnection_2d

def get_parser():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--fold_ind", type=int, default=4,
                        help="1 to 5")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/five-fold',
                        help="the data dir")

    parser.add_argument("--coarse_identifier", type=str,
                        default='DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_'
                                'Adam_lr_0.001_pretrained',
                        help="coarse identifier")

    # parser.add_argument("--coarse_identifier", type=str,
    #                     default='UNet3D_loss_CrossEntropyLoss_Adam_lr_0.001',
    #                     help="coarse identifier")

    parser.add_argument("--model", type=str, default='ResidualUNet2D',
                        help="the model name, UNet2D, ResidualUNet2D, DeepLabv3_plus_skipconnection_2d,"
                             "DeepLabv3_plus_gcn_skipconnection_2d")

    parser.add_argument("--gcn_mode", type=int, default=2, help="0, 1, 2")

    parser.add_argument("--ds_weight", type=float, default=0.3, help="the loss weight for gcn_mode=2.")

    parser.add_argument("--augment", dest='augment', action='store_true',
                        help="whether use augmentation")

    parser.add_argument("--no-augment", dest='augment', action='store_false',
                        help="whether use augmentation")
    parser.set_defaults(augment=False)

    parser.add_argument("--unary", dest='use_unary', action='store_true',
                        help="whether use unary")

    parser.add_argument("--no-unary", dest='use_unary', action='store_false',
                        help="whether use unary")

    parser.set_defaults(use_unary=True)

    parser.add_argument("--pre_trained", dest='pre_trained', action='store_true',
                        help="whether use pre_trained")

    parser.add_argument("--no-pre_trained", dest='pre_trained', action='store_false',
                        help="whether use pre_trained")

    parser.set_defaults(pre_trained=False)

    parser.add_argument("--resume", dest='resume', action='store_true',
                        help="whether use resume")

    parser.add_argument("--no-resume", dest='resume', action='store_false',
                        help="whether use resume")

    parser.set_defaults(resume=False)

    parser.add_argument("--dimension", type=int, default=2,
                        help="the dimension of the image")

    parser.add_argument("--in_channels", type=int, default=128,
                        help="the channels of the input feature map")

    parser.add_argument("--inter_channels", type=int, default=64,
                        help="the channels of the feature maps when calculating the dot product in non-local crf")

    parser.add_argument("--device", type=str, default='cuda:4',
                        help="which gpu to use")

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="Adam or SGD")

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="The initial learning rate")

    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help="The weight decay")

    parser.add_argument('--blur', type=int, default=1,
                        help="The downsample rate for gaussian")

    parser.add_argument('--num_iter', type=int, default=1,
                        help="The number of iterations for CRF")

    parser.add_argument('--trainable', type=bool, default=False,
                        help="Is trainable for the crf?")

    parser.add_argument('--manual_seed', type=int, default=0,
                        help="The manual_seed")

    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        help="The loss function name, CrossEntropyLoss, PixelWiseCrossEntropyLoss")

    parser.add_argument('--eval_metric', type=str, default='DiceCoefficient',
                        help="The eval_metric name, MeanIoU or DiceCoefficient")

    parser.add_argument('--skip_channels', type=list, default=[0],
                        help="The skip_channels in eval_metric")

    parser.add_argument('--pyinn', action='store_true',
                        help="Use pyinn based Cuda implementation"
                             "for message passing.")
    return parser

default_conf = {
    'merge': True,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 1,
    "weight_init": 0.2,

    'convcomp': True, # Compatibility Transform
    'logsoftmax': True,  # use logsoftmax for numerical stability
    'softmax': True,
    'final_softmax': False,

    "pyinn": False
}

def get_default_conf():
    return default_conf.copy()


if __name__ == '__main__':
    num_classes = 20
    parser = get_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    fold_ind = args.fold_ind
    coarse_dir = os.path.join(data_dir, 'coarse')
    fine_dir = os.path.join(data_dir, 'fine')
    data_path = os.path.join(fine_dir, 'in/h5py/fold' + str(fold_ind) + '_data.h5')
    if args.coarse_identifier == 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                 'CrossEntropyLoss_Adam_lr_0.001_pretrained':
        data_path = os.path.join(fine_dir, 'in/h5py/fold' + str(fold_ind) + '_data.h5')
    else:
        data_path = os.path.join(fine_dir, 'in/h5py/fold' + str(fold_ind) + '_data_' + args.coarse_identifier + '.h5')
    foldIndData = np.load(os.path.join(fine_dir, 'split_ind_fold' + str(fold_ind) + '.npz'))
    train_ind = foldIndData['train_ind']
    val_ind = foldIndData['val_ind']
    test_ind = foldIndData['test_ind']

    mrDir = os.path.join(fine_dir, 'in/nii/original_mr')
    maskDir = os.path.join(fine_dir, 'in/nii/mask')
    pre_seg_dir = os.path.join(coarse_dir, 'out', 'fold' + str(fold_ind), args.coarse_identifier)

    if args.coarse_identifier == 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                 'CrossEntropyLoss_Adam_lr_0.001_pretrained':
        fold_dir = os.path.join(fine_dir, 'out', 'fold' + str(fold_ind))
    else:
        fold_dir = os.path.join(fine_dir, 'out', 'fold' + str(fold_ind) + '_' + args.coarse_identifier)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    if args.model == 'non_local_crf':
        identifier = args.model + '_' + args.optimizer + '_lr_' + str(args.learning_rate) + '_num_iter_' + str(
            args.num_iter)
    elif args.model == 'DeepLabv3_plus_gcn_skipconnection_2d':
        if args.gcn_mode == 2:
            identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_ds_weight_' + str(args.ds_weight) + \
                         '_' + args.optimizer + '_lr_' + str(
                args.learning_rate) + '_weight_decay_' + str(
                args.weight_decay)
        else:
            identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_' + args.optimizer + '_lr_' + str(
                args.learning_rate) + '_weight_decay_' + str(
                args.weight_decay)
        if args.use_unary is False:
            identifier = identifier + '_noUnary'
            in_channels = 1
        else:
            in_channels = 21
    else:
        identifier = args.model + '_' + args.optimizer + '_lr_' + str(args.learning_rate) + '_weight_decay_' + str(
            args.weight_decay)
        if args.use_unary is False:
            identifier = identifier + '_noUnary'
            in_channels = 1
        else:
            in_channels = 21
    if args.augment:
        identifier = identifier + '_augment'

    if args.loss != 'CrossEntropyLoss':
        identifier = identifier + '_loss_' + args.loss

    if args.resume:
        identifier = identifier + '_resume'
    elif args.pre_trained:
        identifier = identifier + '_pretrained'

    if args.coarse_identifier == 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                 'CrossEntropyLoss_Adam_lr_0.001_pretrained':
        model_dir = os.path.join(fine_dir, 'model', 'fold' + str(fold_ind), identifier)
    else:
        model_dir = os.path.join(fine_dir, 'model', 'fold' + str(fold_ind) + '_' + args.coarse_identifier, identifier)
    model_path = os.path.join(model_dir, 'best_checkpoint.pytorch')
    out_dir = os.path.join(fold_dir, identifier)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    conf = get_default_conf()
    conf['device'] = args.device
    conf['dimension'] = args.dimension
    conf['in_channels'] = args.in_channels
    conf['inter_channels'] = args.inter_channels
    conf['num_iter'] = args.num_iter
    conf['blur'] = args.blur
    conf['num_iter'] = args.num_iter
    conf['trainable'] = args.trainable
    conf['manual_seed'] = args.manual_seed

    conf['loss'] = {}
    conf['loss']['name'] = args.loss

    conf['eval_metric'] = {}
    conf['eval_metric']['name'] = args.eval_metric
    conf['eval_metric']['skip_channels'] = args.skip_channels


    logger = get_logger('Trainer')

    # Load and log experiment configuration
    logger.info('The configurations: ')
    for k,v in conf.items():
        print('%s: %s' % (k, v))

    manual_seed = conf.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    if args.model == 'UNet2D':
        model = UNet2D(in_channels=in_channels, out_channels=20, final_sigmoid=False, f_maps=32, layer_order='cbr', num_groups=8)
    elif args.model == 'ResidualUNet2D':
        model = ResidualUNet2D(in_channels=in_channels, out_channels=20, final_sigmoid=False, f_maps=32, conv_layer_order='cbr',
                               num_groups=8)
    elif args.model == 'DeepLabv3_plus_skipconnection_2d':
        model = DeepLabv3_plus_skipconnection_2d(nInputChannels=in_channels, n_classes=20, os=16, pretrained=False,
                                                 _print=True, final_sigmoid=False)
    elif args.model == 'DeepLabv3_plus_gcn_skipconnection_2d':
        model = DeepLabv3_plus_gcn_skipconnection_2d(nInputChannels=in_channels, n_classes=20, os=16, pretrained=False,
                                                     _print=True, final_sigmoid=False, hidden_layers=128,
                                                     gcn_mode=args.gcn_mode,
                                                     device=conf['device'])

    # put the model on GPUs
    logger.info(f"Sending the model to '{conf['device']}'")
    model = model.to(conf['device'])
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    try:
        train_data_loader, val_data_loader, test_data_loader, f = load_data(data_path, batch_size=1, shuffle=False)

        model.eval()

        if args.dimension == 2:
            crf_slices = []
            for i, t in enumerate(test_data_loader):
                t = tuple([x.to(conf['device']) for x in t])
                mr, _, unary, _ = t
                output, _, _ = model(unary=unary, img=mr)
                crf_slice = output.to('cpu').detach().numpy() # [batch, class_num, h/2, w]
                crf_slices.append(crf_slice)

            start_ind = 0
            end_ind = 0
            unary_dices = []
            crf_dices = []
            for i in test_ind:
                h_mr_nii = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz'))
                h_mr = h_mr_nii.get_data() # [h, w, d]
                h_pre_seg = nib.load(os.path.join(pre_seg_dir, 'seg_Case' + str(i) + '.nii.gz')).get_data() # [h, w, d]
                h_mask_nii = nib.load(os.path.join(maskDir, 'mask_case' + str(i) + '.nii.gz'))
                h_mask = h_mask_nii.get_data() # [h, w, d]
                h, w, d = h_mr.shape
                end_ind = start_ind + d

                crf_volume = np.concatenate(crf_slices[start_ind : end_ind], axis=0) # [d, class_num, 256, 512]
                crf_volume = crf_volume.transpose((1, 0, 2, 3)) # [class_num, d, 256, 512]
                prob = np.zeros((crf_volume.shape[0], d, h, w), dtype=np.float32) # [class_num, d, h, w]

                # for j in range(crf_volume.shape[0]):
                #     prob[j, :, int(h / 4.):-int(h / 4.), :] = transform.resize(crf_volume[j].astype(np.float), (d, int(h / 2.), w), order=3, mode='constant')

                crf_volume_tensor = torch.from_numpy(crf_volume).to(device=conf['device']) # [class_num, d, 256, 512]
                crf_volume_tensor = crf_volume_tensor.view([1, crf_volume_tensor.shape[0], crf_volume_tensor.shape[1],
                                                            crf_volume_tensor.shape[2], crf_volume_tensor.shape[3]])
                crf_volume_tensor = F.interpolate(crf_volume_tensor, size=(d, int(h / 2.), w), mode='trilinear', align_corners=True)
                crf_volume = crf_volume_tensor.to('cpu').numpy().squeeze()
                prob[:, :, int(h / 4.):-int(h / 4.), :] = crf_volume

                seg = np.argmax(prob, axis=0).transpose((1, 2, 0)) # [h, w, d]
                start_ind = end_ind

                unary_dice = utils.dice_all_class(h_pre_seg, h_mask, class_num=20)
                crf_dice = utils.dice_all_class(seg, h_mask, class_num=20)

                unary_dices.append(unary_dice)
                crf_dices.append(crf_dice)
                print('case %d, unary Dice = %.4f, crf Dice = %.4f' % (i, unary_dice, crf_dice))
                segNii = nib.Nifti1Image(seg.astype(np.uint8), affine=h_mr_nii.affine)
                seg_path = os.path.join(out_dir, 'seg_Case' + str(i) + '.nii.gz')
                nib.save(segNii, seg_path)

            print('mean unary Dice = %.4f, mean crf Dice = %.4f' % (np.mean(unary_dices), np.mean(crf_dices)))
            np.savez(os.path.join(out_dir, 'eval_scores.npz'), unary_dices=unary_dices, crf_dices=crf_dices,
                     mean_unary_dice=np.mean(unary_dices), mean_crf_dice=np.mean(crf_dices))
        elif args.dimension == 3:
            crf_outs = []
            for i, t in enumerate(test_data_loader):
                print(i)
                t = tuple([x.to(conf['device']) for x in t])
                mr, feature, unary, target = t
                output, _ = model(unary=unary, img=mr, col_feats=feature)
                # output = nn.Softmax(dim=1)(output).to('cpu').detach().numpy()  # [1, class_num, 18, 256, 512]
                output = output.to('cpu').detach().numpy() # [1, class_num, 18, 256, 512]
                output = np.squeeze(output) # [class_num, 18, 256, 512]
                torch.cuda.empty_cache()
                crf_outs.append(output)

            count = 0
            unary_dices = []
            crf_dices = []
            for i in test_ind:
                h_mr_nii = nib.load(os.path.join(mrDir, 'Case' + str(i) + '.nii.gz'))
                h_mr = h_mr_nii.get_data()  # [h, w, d]
                h_pre_seg = nib.load(os.path.join(pre_seg_dir, 'seg_Case' + str(i) + '.nii.gz')).get_data()  # [h, w, d]
                h_mask_nii = nib.load(os.path.join(maskDir, 'mask_case' + str(i) + '.nii.gz'))
                h_mask = h_mask_nii.get_data()  # [h, w, d]
                h, w, d = h_mr.shape

                crf_out = crf_outs[count] # [class_num, 18, 256, 512]
                count += 1
                depth = crf_out.shape[1]
                delata = depth - d
                start_d = int(math.ceil(delata / 2.))
                end_d = -int(delata - start_d)
                if end_d == 0:
                    end_d = None
                prob = np.zeros((crf_out.shape[0], d, h, w), dtype=np.float32)  # [class_num, d, h, w]
                for j in range(crf_out.shape[0]):
                    prob[j, :, int(h / 4.):-int(h / 4.), :] = transform.resize(crf_out[j, start_d:end_d, :, :].astype(np.float),
                                                                               (d, int(h / 2.), w), order=3,
                                                                               mode='constant')
                seg = np.argmax(prob, axis=0).transpose((1, 2, 0))  # [h, w, d]

                unary_dice = utils.dice_all_class(h_pre_seg, h_mask, class_num=20)
                crf_dice = utils.dice_all_class(seg, h_mask, class_num=20)

                unary_dices.append(unary_dice)
                crf_dices.append(crf_dice)
                print('case %d, unary Dice = %.4f, crf Dice = %.4f' % (i, unary_dice, crf_dice))
                segNii = nib.Nifti1Image(seg.astype(np.uint8), affine=h_mr_nii.affine)
                seg_path = os.path.join(out_dir, 'seg_Case' + str(i) + '.nii.gz')
                nib.save(segNii, seg_path)

            print('mean unary Dice = %.4f, mean crf Dice = %.4f' % (np.mean(unary_dices), np.mean(crf_dices)))
            np.savez(os.path.join(out_dir, 'eval_scores.npz'), unary_dices=unary_dices, crf_dices=crf_dices,
                     mean_unary_dice=np.mean(unary_dices), mean_crf_dice=np.mean(crf_dices))
    finally:
        f.close()













