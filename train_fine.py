import numpy as np
import h5py
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets.fine_h5 import load_data
import os
import shutil
from networks.utils import get_logger, get_number_of_learnable_parameters
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from networks.losses import get_loss_criterion
from networks.metrics import get_evaluation_metric
import importlib
from networks.trainer_2d import Trainer
from networks.unet_2d import UNet2D, ResidualUNet2D
from networks.deeplab_xception_skipconnection_2d import DeepLabv3_plus_skipconnection_2d
from networks.deeplab_xception_gcn_skipconnection_2d import DeepLabv3_plus_gcn_skipconnection_2d

def get_parser():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--fold_ind", type=int, default=1,
                        help="1 to 5")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/five-fold/fine',
                        help="the data dir")

    parser.add_argument("--coarse_identifier", type=str,
                        default='DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_CrossEntropyLoss_'
                                'Adam_lr_0.001_pretrained',
                        help="coarse identifier")

    parser.add_argument("--model", type=str, default='ResidualUNet2D',
                        help="the model name, UNet2D, ResidualUNet2D, DeepLabv3_plus_skipconnection_2d,"
                             "DeepLabv3_plus_gcn_skipconnection_2d")
    parser.add_argument("--gcn_mode", type=int, default=2, help="0, 1, 2")

    parser.add_argument("--ds_weight", type=float, default=0.3, help="the loss weight for gcn_mode=2.")

    parser.add_argument("--augment", dest='augment', action='store_true',
                        help="whether use augmentation")

    parser.add_argument("--no-augment", dest='augment', action='store_false',
                        help="whether use augmentation")
    parser.set_defaults(augment=True)

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

    parser.add_argument("--epochs", type=int, default=100,
                        help="max number of epochs")

    parser.add_argument("--iters", type=int, default=1000000,
                        help="max number of iterations for training")

    parser.add_argument("--eval_score_higher_is_better", type=bool, default=True,
                        help="model with higher eval score is considered better")

    parser.add_argument("--device", type=str, default='cuda:0',
                        help="which gpu to use")

    parser.add_argument('--batch_size', type=int, default=8,
                        help="The batch size")

    parser.add_argument('--manual_seed', type=int, default=0,
                        help="The manual_seed")

    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        help="The loss function name, CrossEntropyLoss, PixelWiseCrossEntropyLoss")

    parser.add_argument('--eval_metric', type=str, default='DiceCoefficient',
                        help="The eval_metric name, MeanIoU or DiceCoefficient")

    parser.add_argument('--skip_channels', type=list, default=None,
                        help="The skip_channels in eval_metric")

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="Adam or SGD")

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="The initial learning rate")

    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help="The weight decay")

    parser.add_argument('--momentum', type=float, default=0.9,
                        help="the momentum for the optimizer")

    parser.add_argument('--nesterov', type=bool, default=False,
                        help="the nesterov for the optimizer")

    parser.add_argument('--gamma', type=float, default=0.2,
                        help="The gamma for the MultiStepLR learning scheduler")

    return parser

default_conf = {
    "pyinn": False,

    'transformer':{
        'train':{
            'raw': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                {'name': 'ElasticDeformation', 'spline_order': 3},
                {'name': 'RandomContrast'}
            ],
            'label': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'nearest'},
                {'name': 'ElasticDeformation', 'spline_order': 0}
            ],
            'unary': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                {'name': 'ElasticDeformation', 'spline_order': 3}
            ],
            'weight': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                {'name': 'ElasticDeformation', 'spline_order': 3}
            ]
        },
        'test':{
            'raw': None,
            'label': None,
            'unary': None,
            'weight': None
        }
    }
}

def get_default_conf():
    return default_conf.copy()

def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    momentum = optimizer_config.get('momentum', 0.0)
    nesterov = optimizer_config.get('nesterov', False)
    assert optimizer_name in ['Adam', 'SGD'], 'the optimizer name should be Adam or SGD'
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                              nesterov=nesterov)
        return optimizer

def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)

def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, train_loader, val_loader, logger):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)

    if resume is not None:
        # continue training from a given checkpoint
        return Trainer.from_checkpoint(resume, model,
                                             optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, train_loader, val_loader,
                                             logger=logger)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return Trainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, device=config['device'],
                                             train_loader=train_loader, val_loader=val_loader,
                                             max_num_epochs=trainer_config['epochs'],
                                             batch_size=trainer_config['batch_size'],
                                             max_num_iterations=trainer_config['iters'],
                                             validate_after_iters=trainer_config['validate_after_iters'],
                                             log_after_iters=trainer_config['log_after_iters'],
                                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                                             logger=logger)
    else:
        # start training from scratch
        return Trainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                             config['device'], train_loader, val_loader, trainer_config['checkpoint_dir'],
                             max_num_epochs=trainer_config['epochs'],
                             batch_size=trainer_config['batch_size'],
                             max_num_iterations=trainer_config['iters'],
                             validate_after_iters=trainer_config['validate_after_iters'],
                             log_after_iters=trainer_config['log_after_iters'],
                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                             logger=logger)

def main():
    parser = get_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    fold_ind = args.fold_ind
    batch_size = args.batch_size

    if args.coarse_identifier == 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                 'CrossEntropyLoss_Adam_lr_0.001_pretrained':
        data_path = os.path.join(data_dir, 'in/h5py/fold' + str(fold_ind) + '_data.h5')
    else:
        data_path = os.path.join(data_dir, 'in/h5py/fold' + str(fold_ind) + '_data_' + args.coarse_identifier + '.h5')

    if args.coarse_identifier == 'DeepLabv3_plus_gcn_skipconnection_3d_gcn_mode_2_ds_weight_0.3_loss_' \
                                 'CrossEntropyLoss_Adam_lr_0.001_pretrained':
        fold_dir = os.path.join(data_dir, 'model', 'fold' + str(fold_ind))
    else:
        fold_dir = os.path.join(data_dir, 'model', 'fold' + str(fold_ind) + '_' + args.coarse_identifier)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    if args.model == 'non_local_crf':
        identifier = args.model + '_' + args.optimizer + '_lr_' + str(args.learning_rate) + '_num_iter_' + str(
            args.num_iter)
    elif args.model == 'DeepLabv3_plus_gcn_skipconnection_2d':
        if args.gcn_mode == 2:
            identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_ds_weight_' + str(args.ds_weight) +\
                         '_' + args.optimizer + '_lr_' + str(
                args.learning_rate) + '_weight_decay_' + str(
                args.weight_decay)
        else:
            identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_' + args.optimizer + '_lr_' + str(args.learning_rate) + '_weight_decay_' + str(
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

    conf = get_default_conf()
    conf['device'] = args.device
    conf['manual_seed'] = args.manual_seed

    conf['loss'] = {}
    conf['loss']['name'] = args.loss

    conf['eval_metric'] = {}
    conf['eval_metric']['name'] = args.eval_metric
    conf['eval_metric']['skip_channels'] = args.skip_channels

    conf['optimizer'] = {}
    conf['optimizer']['name'] = args.optimizer
    conf['optimizer']['learning_rate'] = args.learning_rate
    conf['optimizer']['weight_decay'] = args.weight_decay
    conf['optimizer']['momentum'] = args.momentum
    conf['optimizer']['nesterov'] = args.nesterov

    conf['lr_scheduler'] = {}
    conf['lr_scheduler']['name'] = 'MultiStepLR'
    conf['lr_scheduler']['milestones'] = [args.epochs // 3, args.epochs // 1.5]
    conf['lr_scheduler']['gamma'] = args.gamma

    conf['trainer'] = {}

    conf['trainer']['batch_size'] = batch_size
    conf['trainer']['epochs'] = args.epochs
    conf['trainer']['iters'] = args.iters
    conf['trainer']['eval_score_higher_is_better'] = args.eval_score_higher_is_better
    conf['trainer']['ds_weight'] = args.ds_weight

    if args.loss == 'PixelWiseCrossEntropyLoss':
        return_weight = True
    else:
        return_weight = False

    if args.resume:
        identifier = identifier + '_resume'
        conf['trainer']['resume'] = os.path.join(fold_dir, identifier,'best_checkpoint.pytorch')
        if not os.path.exists(os.path.join(fold_dir, identifier)):
            src = os.path.join(fold_dir, 'DeepLabv3_plus_skipconnection_2d' + '_Adam_lr_' + str(args.learning_rate) +
                                '_weight_decay_' + str(args.weight_decay) + '_noUnary_augment')
            shutil.copytree(src=src, dst=os.path.join(fold_dir, identifier))
    elif args.pre_trained:
        identifier = identifier + '_pretrained'
        conf['trainer']['pre_trained'] = os.path.join(fold_dir, identifier, 'best_checkpoint.pytorch')
        if not os.path.exists(os.path.join(fold_dir, identifier)):
            src = os.path.join(fold_dir, 'DeepLabv3_plus_skipconnection_2d' + '_Adam_lr_' + str(args.learning_rate) +
                               '_weight_decay_' + str(args.weight_decay) + '_noUnary_augment')
            shutil.copytree(src=src, dst=os.path.join(fold_dir, identifier))
    checkpoint_dir = os.path.join(fold_dir, identifier)
    conf['trainer']['checkpoint_dir'] = checkpoint_dir


    logger = get_logger('Trainer')

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
    if args.model == 'UNet2D':
        model = UNet2D(in_channels=in_channels, out_channels=20, final_sigmoid=False, f_maps=32, layer_order='cbr',
                       num_groups=8)
    elif args.model == 'ResidualUNet2D':
        model = ResidualUNet2D(in_channels=in_channels, out_channels=20, final_sigmoid=False, f_maps=32,
                               conv_layer_order='cbr', num_groups=8)
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
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(conf)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(conf)

    try:
        if args.augment:
            train_data_loader, val_data_loader, test_data_loader, f = load_data(data_path, batch_size=batch_size,
                                                                                transformer_config=conf['transformer'],
                                                                                return_weight=return_weight)
        else:
            train_data_loader, val_data_loader, test_data_loader, f = load_data(data_path, batch_size=batch_size,
                                                                                return_weight=return_weight)
        conf['trainer']['validate_after_iters'] = len(train_data_loader)
        conf['trainer']['log_after_iters'] = len(train_data_loader)
        # Create the optimizer
        optimizer = _create_optimizer(conf, model)

        # Create learning rate adjustment strategy
        lr_scheduler = _create_lr_scheduler(conf, optimizer)

        # Create model trainer
        trainer = _create_trainer(conf, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loss_criterion=loss_criterion, eval_criterion=eval_criterion,
                                  train_loader=train_data_loader,
                                  val_loader=val_data_loader,
                                  logger=logger)
        # Start training
        trainer.fit()
    finally:
        f.close()
if __name__ == '__main__':
    main()











