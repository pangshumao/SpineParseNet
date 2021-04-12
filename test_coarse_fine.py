import torch
import os
from networks.metrics import get_evaluation_metric
from networks.utils import get_logger
from networks.utils import get_number_of_learnable_parameters
from networks.deeplab_xception_gcn_skipconnection_3d import DeepLabv3_plus_gcn_skipconnection_3d
from networks.deeplab_xception_skipconnection_3d import DeepLabv3_plus_skipconnection_3d
from networks.model import ResidualUNet3D, UNet3D
from networks.unet_2d import UNet2D, ResidualUNet2D
from networks import graph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import nibabel as nib
from skimage import transform
import math
from torch.nn import functional as F
from networks import utils
import time

def get_graph(device):
    spine_adj = graph.preprocess_adj(graph.spine_graph)
    spine_adj_ = torch.from_numpy(spine_adj).float()
    spine_adj = spine_adj_.unsqueeze(0).unsqueeze(0).to(device)
    return spine_adj

def transform_volume(raw, mean=None, std=None, is_crop=True, is_expand_dim=True, scale=[0.5, 0.5, 1.0], interpolation='cubic'):
    '''

    :param raw: a numpy array with a shape of [h, w, d]
    :param mean:
    :param std:
    :param is_crop:
    :param is_expand_dim:
    :param scale:
    :param interpolation: 'cubic' or 'nearest'
    :return: raw - if is_expand_dim is True, the shape of raw is [1, 1, D, H, W], otherwise, [D, H, W]
             crop_h_start - the start coordinate of h when crop the volume
             crop_h_end - the end coordinate of h when crop the volume
    '''
    assert interpolation in ['nearest', 'cubic']
    raw = raw.transpose((2, 0, 1)) # [d, h, w]
    crop_h_start = 0
    crop_h_end = -1
    if is_crop is True:
        d, h, w = raw.shape
        crop_h_start = int(h / 4.)
        crop_h_end = -int(h / 4.)
        raw = raw[:, crop_h_start : crop_h_end, :]

    d, h, w = raw.shape
    if interpolation == 'cubic':
        raw = transform.resize(raw.astype(np.float), (int(d * scale[2]), int(h * scale[0]), int(w * scale[1])), order=3,
                               mode='constant')
    elif interpolation == 'nearest':
        raw = transform.resize(raw.astype(np.float), (int(d * scale[2]), int(h * scale[0]), int(w * scale[1])), order=0,
                               anti_aliasing=False, mode='constant')
    if mean is not None:
        raw -= mean
        raw /= std
    if is_expand_dim is True:
        raw = np.expand_dims(raw, axis=0)
        raw = np.expand_dims(raw, axis=0)
    return raw, crop_h_start, crop_h_end

def inv_transform_volume(raw, crop_h_start, crop_h_end, original_shape, transform_scale=[0.5, 0.5, 1.0],
                         interpolation='cubic'):
    '''

    :param raw: a numpy array with a shape of [h, w, d] or [1, c, d, h, w]
    :param crop_h_start:
    :param crop_h_end:
    :param original_shape: [H, W, D]
    :param transform_scale:
    :param interpolation: 'cubic' or 'nearest'
    :return: a numpy array with a shape of [H, W, D] or [1, c, D, H, W]
    '''
    assert interpolation in ['cubic', 'nearest']
    assert raw.ndim in [3, 5]
    if raw.ndim == 3:
        h,w,d = raw.shape
        raw = raw.transpose((2,0,1)) # [d, h, w]
        if isinstance(original_shape, list):
            original_shape = tuple(original_shape)
        out_volume = np.zeros(original_shape, dtype=raw.dtype)
        if interpolation == 'cubic':
            raw = transform.resize(raw.astype(np.float), (int(d / transform_scale[2]), int(h / transform_scale[0]),
                                                          int(w / transform_scale[1])), order=3, mode='constant')
        elif interpolation == 'nearest':
            raw = transform.resize(raw.astype(np.float), (int(d / transform_scale[2]), int(h / transform_scale[0]),
                                                          int(w / transform_scale[1])), order=0, anti_aliasing=False,
                                   mode='constant')
        out_volume[crop_h_start : crop_h_end, :, :] = raw.transpose((1,2,0))
        return out_volume
    elif raw.ndim == 5:
        n, c, d, h, w = raw.shape
        raw = np.squeeze(raw, axis=0) # [c, d, h, w]
        raw = raw.transpose((0, 2, 3, 1)) # [c, h, w, d]

        out_volume = np.zeros((c, original_shape[0], original_shape[1], original_shape[2]), dtype=raw.dtype) # [c, H, W, D]
        for i in range(0, c):
            if interpolation == 'cubic':
                resize_raw = transform.resize(raw[i].astype(np.float), (int(h / transform_scale[0]),
                                               int(w / transform_scale[1]), int(d / transform_scale[2])), order=3, mode='constant')
                out_volume[i, crop_h_start:crop_h_end, :, :] = resize_raw
            elif interpolation == 'nearest':
                resize_raw = transform.resize(raw[i].astype(np.float), (int(h / transform_scale[0]),
                                                                        int(w / transform_scale[1]),
                                                                        int(d / transform_scale[2])), order=0,
                                              anti_aliasing=False, mode='constant')
                out_volume[i, crop_h_start:crop_h_end, :, :] = resize_raw
        out_volume = out_volume.transpose((0, 3, 1, 2)) # [c, D, H, W]
        out_volume = np.expand_dims(out_volume, axis=0) # [1, c, D, H, W]
        return out_volume

def inv_transform_volume_tensor(input, crop_h_start, crop_h_end, original_shape, is_crop_d=False, start_d=0, end_d=0):
    '''

    :param input: a tensor with shape of [b, C, d, h, w]
    :param crop_h_start:
    :param crop_h_end:
    :param original_shape: [H, W, D]
    :param is_crop_d:
    :param start_d:
    :param end_d:
    :return: a tensor with shape of [b, C, D, H, W]
    '''
    H, W, D = original_shape
    input = F.interpolate(input, size=[D, int(H / 2.), W], mode='trilinear', align_corners=True).to('cpu')  # [b, C, D, H/2, W]
    out = torch.zeros((input.shape[0], input.shape[1], D, H, W), dtype=input.dtype)
    out[:, :, :, crop_h_start : crop_h_end, :] = input

    if is_crop_d is True:
        if end_d == 0:
            return out[:, :, start_d:, :, :]
        else:
            return out[:, :, start_d:-end_d, :, :]
    else:
        return out


def pad_d(volume, new_d=15):
    '''

    :param volume: a numpy array with shape of [d, h, w]
    :param new_d:
    :return:
    '''
    d, h, w = volume.shape
    new_volume = np.zeros(shape=(new_d, h, w), dtype=volume.dtype)
    delata = new_d - d
    start_ind = int(math.ceil(delata / 2.))
    end_ind = int(delata - start_ind)
    if end_ind == 0:
        new_volume[start_ind:, :, :] = volume
    else:
        new_volume[start_ind:-end_ind, :, :] = volume
    return new_volume, start_ind, end_ind

def inv_pad_d(volume, start_ind, end_ind):
    '''

    :param volume: a numpy array with shape of [d, h, w] or [c, d, h, w] or [n, c, d, h, w]
    :param start_ind:
    :param end_ind:
    :return:
    '''
    if volume.ndim == 3:
        if end_ind == 0:
            return volume[start_ind:, :, :]
        else:
            return volume[start_ind:-end_ind, :, :]
    elif volume.ndim == 4:
        if end_ind == 0:
            return volume[:, start_ind:, :, :]
        else:
            return volume[:, start_ind:-end_ind, :, :]
    elif volume.ndim == 5:
        if end_ind == 0:
            return volume[:, :, start_ind:, :, :]
        else:
            return volume[:, :, start_ind:-end_ind, :, :]

def inv_pad_d_tensor(volume, start_ind, end_ind):
    '''

    :param volume: a tensor with shape of [d, h, w] or [c, d, h, w] or [n, c, d, h, w]
    :param start_ind:
    :param end_ind:
    :return:
    '''
    if volume.dim() == 3:
        if end_ind == 0:
            return volume[start_ind:, :, :]
        else:
            return volume[start_ind:-end_ind, :, :]
    elif volume.dim() == 4:
        if end_ind == 0:
            return volume[:, start_ind:, :, :]
        else:
            return volume[:, start_ind:-end_ind, :, :]
    elif volume.dim() == 5:
        if end_ind == 0:
            return volume[:, :, start_ind:, :, :]
        else:
            return volume[:, :, start_ind:-end_ind, :, :]

def get_parser():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--fold_ind", type=int, default=1,
                        help="1 to 5")

    parser.add_argument("--model", type=str, default='DeepLabv3_plus_gcn_skipconnection_3d',
                        help="the model name, DeepLabv3_plus_skipconnection_3d, "
                             "DeepLabv3_plus_gcn_skipconnection_3d,"
                             "UNet3D,"
                             "ResidualUNet3D")

    parser.add_argument("--gcn_mode", type=int, default=2,
                        help="the mode for fea2graph and graph2fea, only available for gcn. 0, 1, 2")

    parser.add_argument("--ds_weight", type=float, default=0.3,
                        help="The deep supervision weight used in fea2graph when gcn_mode is 2.")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/five-fold',
                        help="the data dir")

    parser.add_argument("--resume", dest='resume', action='store_true',
                        help="the training will be resumed from that checkpoint.")

    parser.add_argument("--no-resume", dest='resume', action='store_false',
                        help="the training will not be resumed from that checkpoint.")

    parser.set_defaults(resume=False)

    parser.add_argument("--pre_trained", dest='pre_trained', action='store_true',
                        help="use pretrained the model.")

    parser.add_argument("--no-pre_trained", dest='pre_trained', action='store_false',
                        help="without using pretrained the model.")

    parser.set_defaults(pre_trained=True)

    parser.add_argument("--device", type=str, default='cuda:0',
                        help="which gpu to use")

    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        help="The loss function name, FPFNLoss, CrossEntropyLoss")

    parser.add_argument('--lamda', type=float, default=0.1,
                        help="For FPFNLoss")

    parser.add_argument('--eval_metric', type=str, default='DiceCoefficient',
                        help="The eval_metric name, MeanIoU or DiceCoefficient")

    parser.add_argument('--skip_channels', type=list, default=[0],
                        help="The skip_channels in eval_metric")

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="Adam or SGD")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="The initial learning rate")

    parser.add_argument('--seed', type=int, default=0,
                        help="The manual seed")

    return parser

default_conf = {
    'manual_seed': 0,
    'model': {
        'in_channels': 1,
        'out_channels': 20,
        'hidden_layers': 128,
        'layer_order': 'crg',
        'f_maps': 16,
        'num_groups': 1,
        'final_sigmoid': False
    },
    'transformer':{
        'train':{
            'raw': [
                {'name': 'RandomRotate', 'axes': [[2, 1]], 'angle_spectrum': 15, 'mode': 'reflect'},
                {'name': 'ElasticDeformation', 'spline_order': 3},
                {'name': 'RandomContrast'}
            ],
            'label': [
                {'name': 'RandomRotate', 'axes': [[2, 1]], 'angle_spectrum': 15, 'mode': 'reflect'},
                {'name': 'ElasticDeformation', 'spline_order': 0}
            ],
            'weight': [
                {'name': 'RandomRotate', 'axes': [[2, 1]], 'angle_spectrum': 15, 'mode': 'reflect'},
                {'name': 'ElasticDeformation', 'spline_order': 3}
            ]
        },
        'test':{
            'raw': None,
            'label': None,
            'weight': None
        }
    },
    'optimizer':{
        'momentum': 0.9,
        'nesterov': False,
        'weight_decay': 0.0001
    },
    'lr_scheduler':{
        'name': 'MultiStepLR',
        'gamma': 0.2
    },
    'trainer':{
        'eval_score_higher_is_better':True,
        'iters': 100000000
    }


}

def get_default_conf():
    return default_conf.copy()

def main():
    start_time = time.time()
    mean = 466.0
    std = 379.0
    parser = get_parser()
    args = parser.parse_args()
    foldInd = args.fold_ind
    dataDir = args.data_dir
    foldIndData = np.load(os.path.join(dataDir, 'coarse', 'split_ind_fold' + str(foldInd) + '.npz'))
    train_ind = foldIndData['train_ind']
    val_ind = foldIndData['val_ind']
    test_ind = foldIndData['test_ind']

    conf = get_default_conf()
    conf['manual_seed'] = args.seed
    conf['device'] = args.device

    conf['model']['name'] = args.model
    conf['model']['gcn_mode'] = args.gcn_mode

    conf['loss'] = {}
    conf['loss']['name'] = args.loss
    conf['loss']['lamda'] = args.lamda

    conf['eval_metric'] = {}
    conf['eval_metric']['name'] = args.eval_metric
    conf['eval_metric']['skip_channels'] = args.skip_channels

    conf['optimizer']['name'] = args.optimizer
    conf['optimizer']['learning_rate'] = args.learning_rate

    conf['trainer']['resume'] = args.resume
    conf['trainer']['pre_trained'] = args.pre_trained
    conf['trainer']['ds_weight'] = args.ds_weight

    train_coarse_foldDir = os.path.join(dataDir, 'coarse', 'model', 'fold' + str(foldInd))
    train_fine_foldDir = os.path.join(dataDir, 'fine', 'model', 'fold' + str(foldInd))

    if args.loss == 'FPFNLoss':
        return_weight = True
        if 'gcn' in args.model:
            if args.gcn_mode == 2:
                identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_ds_weight_' + str(args.ds_weight) + \
                             '_loss_' + args.loss + '_lamda_' + str(
                    args.lamda) + '_' + args.optimizer + '_lr_' + str(args.learning_rate)
            else:
                identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_loss_' + args.loss + '_lamda_' + str(
                    args.lamda) \
                             + '_' + args.optimizer + '_lr_' + str(args.learning_rate)
        else:
            identifier = args.model + '_loss_' + args.loss + '_lamda_' + str(
                args.lamda) + '_' + args.optimizer + '_lr_' + str(args.learning_rate)
    elif args.loss == 'PixelWiseCrossEntropyLoss':
        return_weight = True
        if 'gcn' in args.model:
            if args.gcn_mode == 2:
                identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_ds_weight_' + str(args.ds_weight) + \
                             '_loss_' + args.loss + '_' + args.optimizer + \
                             '_lr_' + str(args.learning_rate)
            else:
                identifier = args.model + '_gcn_mode_' + str(
                    args.gcn_mode) + '_loss_' + args.loss + '_' + args.optimizer + \
                             '_lr_' + str(args.learning_rate)
        else:
            identifier = args.model + '_loss_' + args.loss + '_' + args.optimizer + '_lr_' + str(args.learning_rate)
    else:
        return_weight = False
        if 'gcn' in args.model:
            if args.gcn_mode == 2:
                identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_ds_weight_' + str(args.ds_weight) + \
                             '_loss_' + args.loss + '_' + args.optimizer + \
                             '_lr_' + str(args.learning_rate)
            else:
                identifier = args.model + '_gcn_mode_' + str(
                    args.gcn_mode) + '_loss_' + args.loss + '_' + args.optimizer + \
                             '_lr_' + str(args.learning_rate)
        else:
            identifier = args.model + '_loss_' + args.loss + '_' + args.optimizer + '_lr_' + str(args.learning_rate)

    if args.seed != 0:
        identifier = identifier + '_seed_' + str(args.seed)

    if args.resume:
        identifier = identifier + '_resume'
    elif args.pre_trained:
        identifier = identifier + '_pretrained'

    fine_identifier = 'ResidualUNet2D_Adam_lr_0.0001_weight_decay_0.0001_augment'
    out_dir = os.path.join(dataDir, 'coarse_fine', 'out', 'fold' + str(foldInd), fine_identifier)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    coarse_checkpoint_dir = os.path.join(train_coarse_foldDir, identifier)
    fine_checkpoint_dir = os.path.join(train_fine_foldDir, fine_identifier)
    coarse_model_path = os.path.join(coarse_checkpoint_dir, 'best_checkpoint.pytorch')
    fine_model_path = os.path.join(fine_checkpoint_dir, 'best_checkpoint.pytorch')
    # model_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')

    # conf['trainer']['checkpoint_dir'] = coarse_checkpoint_dir


    # Create main logger
    logger = get_logger('UNet3DTrainer')

    # Load and log experiment configuration
    logger.info('The configurations: ')
    for k, v in conf.items():
        print('%s: %s' % (k, v))

    manual_seed = conf.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    if args.model == 'DeepLabv3_plus_skipconnection_3d':
        coarse_model = DeepLabv3_plus_skipconnection_3d(nInputChannels=conf['model']['in_channels'],
                                                 n_classes=conf['model']['out_channels'],
                                                 os=16, pretrained=False, _print=True,
                                                 final_sigmoid=conf['model']['final_sigmoid'],
                                                 normalization='bn',
                                                 num_groups=8)
    elif args.model == 'DeepLabv3_plus_gcn_skipconnection_3d':
        coarse_model = DeepLabv3_plus_gcn_skipconnection_3d(nInputChannels=conf['model']['in_channels'], n_classes=conf['model']['out_channels'],
                                  os=16, pretrained=False, _print=True, final_sigmoid=conf['model']['final_sigmoid'],
                                hidden_layers=conf['model']['hidden_layers'], device=conf['device'],
                                                     gcn_mode=conf['model']['gcn_mode'])
    elif args.model == 'UNet3D':
        coarse_model = UNet3D(in_channels=conf['model']['in_channels'], out_channels=conf['model']['out_channels'],
                       final_sigmoid=conf['model']['final_sigmoid'], f_maps=32, layer_order='cbr')
    elif args.model == 'ResidualUNet3D':
        coarse_model = ResidualUNet3D(in_channels=conf['model']['in_channels'], out_channels=conf['model']['out_channels'],
                               final_sigmoid=conf['model']['final_sigmoid'], f_maps=32, conv_layer_order='cbr')

    utils.load_checkpoint(coarse_model_path, coarse_model)
    # put the model on GPUs
    logger.info(f"Sending the model to '{conf['device']}'")
    coarse_model = coarse_model.to(conf['device'])
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(coarse_model)}')


    fine_model = ResidualUNet2D(in_channels=21, out_channels=20, final_sigmoid=False, f_maps=32, conv_layer_order='cbr',
                               num_groups=8)

    utils.load_checkpoint(fine_model_path, fine_model)
    fine_model = fine_model.to(conf['device'])


    # Create evaluation metric
    eval_criterion = get_evaluation_metric(conf)
    coarse_model.eval()
    fine_model.eval()
    eval_scores = []
    for i in test_ind:
        raw_path = os.path.join(dataDir, 'coarse', 'in', 'nii', 'original_mr', 'Case' + str(i) + '.nii.gz')
        label_path = os.path.join(dataDir, 'coarse', 'in', 'nii', 'mask', 'mask_case' + str(i) + '.nii.gz')
        raw_nii = nib.load(raw_path)
        raw = raw_nii.get_data()
        H, W, D = raw.shape # [H, W, D]
        transform_scale = [128./(H/2.), 256./W, 1.0]
        transform_raw, crop_h_start, crop_h_end = transform_volume(raw, is_crop=True, is_expand_dim=False,
                                                                   scale=transform_scale, interpolation='cubic')

        fine_transform_scale = [256. / (H / 2.), 512. / W, 1.0]
        fine_transform_raw, fine_crop_h_start, fine_crop_h_end = transform_volume(raw, is_crop=True, is_expand_dim=False,
                                                                 scale=fine_transform_scale, interpolation='cubic')
        flag = False
        new_d = 18
        if D < new_d:
            flag = True
            transform_raw, start_d, end_d = pad_d(transform_raw, new_d=new_d)
        original_shape = [H, W, D]

        transform_raw -= mean
        transform_raw /= std
        transform_raw = np.expand_dims(transform_raw, axis=0)
        transform_raw = np.expand_dims(transform_raw, axis=0)

        transform_raw = torch.from_numpy(transform_raw.astype(np.float32))

        fine_transform_raw -= mean
        fine_transform_raw /= std

        label_nii = nib.load(label_path)
        label = label_nii.get_data() # numpy
        label = label.transpose((2, 0, 1)) # [D, H, W]

        label = np.expand_dims(label, axis=0) # [1, D, H, W]
        label_tensor = torch.from_numpy(label.astype(np.int64))

        with torch.no_grad():
            coarse_input = transform_raw.to(conf['device'])
            logit = coarse_model(coarse_input)[1] # [1, 20, 18, 256, 128]

            # prob, logit, feature = model(data, return_feat=True)  # batch, classes, d, h, w
            if flag:
                logit = inv_pad_d_tensor(logit, start_d, end_d) # [1, 20, D, 256, 128]

            fine_prob_volume = torch.zeros((1, logit.shape[1], fine_transform_raw.shape[0], fine_transform_raw.shape[1],
                                            fine_transform_raw.shape[2]), dtype=torch.float32, device=conf['device'])
            for j in range(fine_transform_raw.shape[0]):
                fine_mr = fine_transform_raw[j]
                fine_mr = np.expand_dims(fine_mr, axis=0)
                fine_mr = np.expand_dims(fine_mr, axis=0)
                fine_mr = torch.from_numpy(fine_mr.astype(np.float32)).to(conf['device'])
                unary = logit[:, :, j, :, :]

                fine_prob = fine_model(img=fine_mr, unary=unary)[0]
                fine_prob_volume[:, :, j, :, :] = fine_prob

                # temp = fine_prob.to('cpu').numpy()
                # plt.imshow(temp[0, 1, :, :])
                # plt.show()

            fine_prob_volume = inv_transform_volume_tensor(fine_prob_volume, fine_crop_h_start, fine_crop_h_end,
                                        original_shape=original_shape, is_crop_d=False)

            eval_score = eval_criterion(fine_prob_volume, label_tensor)
            fine_prob_volume = fine_prob_volume.numpy()

            if args.eval_metric == 'DiceCoefficient':
                print('Case %d, dice = %f' % (i, eval_score))
            elif args.eval_metric == 'MeanIoU':
                print('Case %d, IoU = %f' % (i, eval_score))

            eval_scores.append(eval_score)

            prob = np.squeeze(fine_prob_volume, axis=0) # numpy, [C, D, H, W]
            seg = np.argmax(prob, axis=0).astype(np.uint8) # numpy, [D, H, W]
            seg = seg.transpose((1,2,0)) # numpy, [H, W, D]

            path_split = raw_path.split(os.sep)
            seg_path = os.sep.join([out_dir, 'seg_' + path_split[-1]])


            segNii = nib.Nifti1Image(seg.astype(np.uint8), affine=raw_nii.affine)
            nib.save(segNii, seg_path)
    if args.eval_metric == 'DiceCoefficient':
        print('mean dice = %f' % np.mean(eval_scores))
        np.savez(os.path.join(out_dir, 'eval_scores.npz'), dice=eval_scores, mean_dice=np.mean(eval_scores))
    elif args.eval_metric == 'MeanIoU':
        print('mean IOU = %f' % np.mean(eval_scores))
        np.savez(os.path.join(out_dir, 'eval_scores.npz'), iou=eval_scores, mean_iou=np.mean(eval_scores))

    end_time = time.time()
    total_time = end_time - start_time
    mean_time = total_time / test_ind.size
    print('mean time for segmenting one volume: %.2f seconds' % mean_time)

if __name__ == '__main__':
    main()
