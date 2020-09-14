import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import importlib
import os
import shutil
from networks.losses import get_loss_criterion
from networks.metrics import get_evaluation_metric
from networks.model import get_model
from networks.trainer import UNet3DTrainer
from networks.utils import get_logger
from networks.utils import get_number_of_learnable_parameters
from networks.deeplab_xception_gcn_skipconnection_3d_modelparallel import DeepLabv3_plus_gcn_skipconnection_3d
from networks.deeplab_xception_skipconnection_3d_modelparallel import DeepLabv3_plus_skipconnection_3d
from networks.model import ResidualUNet3D, UNet3D
from networks import graph
from datasets.coarse_h5 import load_data
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, logger):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)

    if resume is not None:
        # continue training from a given checkpoint
        return UNet3DTrainer.from_checkpoint(resume, model,
                                             optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, loaders,
                                             logger=logger,
                                             ds_weight=trainer_config['ds_weight'])
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return UNet3DTrainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, device=config['device'], loaders=loaders,
                                             max_num_epochs=trainer_config['epochs'],
                                             max_num_iterations=trainer_config['iters'],
                                             validate_after_iters=trainer_config['validate_after_iters'],
                                             log_after_iters=trainer_config['log_after_iters'],
                                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                                             logger=logger,
                                             ds_weight=trainer_config['ds_weight'])
    else:
        # start training from scratch
        return UNet3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                             config['device'], loaders, trainer_config['checkpoint_dir'],
                             max_num_epochs=trainer_config['epochs'],
                             max_num_iterations=trainer_config['iters'],
                             validate_after_iters=trainer_config['validate_after_iters'],
                             log_after_iters=trainer_config['log_after_iters'],
                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                             logger=logger,
                             ds_weight=trainer_config['ds_weight'])


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

def get_graph(device):
    spine_adj = graph.preprocess_adj(graph.spine_graph)
    spine_adj_ = torch.from_numpy(spine_adj).float()
    spine_adj = spine_adj_.unsqueeze(0).unsqueeze(0).to(device)
    return spine_adj

def get_parser():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--fold_ind", type=int, default=1,
                        help="1 to 5")

    parser.add_argument("--model", type=str, default='DeepLabv3_plus_skipconnection_3d',
                        help="the model name, DeepLabv3_plus_skipconnection_3d, "
                             "DeepLabv3_plus_gcn_skipconnection_3d,"
                             "UNet3D,"
                             "ResidualUNet3D")

    parser.add_argument("--gcn_mode", type=int, default=2,
                        help="the mode for fea2graph and graph2fea, only available for gcn. 0, 1, 2")

    parser.add_argument("--ds_weight", type=float, default=0.3,
                        help="The deep supervision weight used in fea2graph when gcn_mode is 2.")

    parser.add_argument("--data_dir", type=str, default='/public/pangshumao/data/five-fold/high_resolution',
                        help="the data dir")

    parser.add_argument("--resume", type=bool, default=False,
                        help="path to latest checkpoint; if provided the training will be resumed from that checkpoint")

    parser.add_argument("--pre_trained", type=bool, default=False,
                        help="similar to resume except that the optimizer is from scratch")

    parser.add_argument("--validate_after_iters", type=int, default=168 // 2,
                        help="how many iterations between validations, 168/batch_size.")

    parser.add_argument("--log_after_iters", type=int, default=168 // 2,
                        help="how many iterations between tensorboard logging, 168/batch_size.")

    parser.add_argument("--epochs", type=int, default=100,
                        help="max number of epochs")

    # parser.add_argument("--device", type=str, default='cuda:0',
    #                     help="which gpu to use")

    parser.add_argument("--device", type=str, default='cuda:0', nargs='+',
                        help="which gpus to use")

    parser.add_argument('--batch_size', type=int, default=2,
                        help="The batch size")

    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        help="The loss function name, FPFNLoss, CrossEntropyLoss, PixelWiseCrossEntropyLoss.")

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
    parser = get_parser()
    args = parser.parse_args()
    foldInd = args.fold_ind
    dataDir = args.data_dir
    filePath = os.path.join(dataDir, 'in', 'h5py', 'data_fold' + str(foldInd) + '.h5')
    batch_size = args.batch_size

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

    conf['lr_scheduler']['milestones'] = [args.epochs // 3, args.epochs // 1.5]

    # conf['trainer']['resume'] = args.resume
    # conf['trainer']['pre_trained'] = args.pre_trained
    conf['trainer']['validate_after_iters'] = args.validate_after_iters
    conf['trainer']['log_after_iters'] = args.log_after_iters
    conf['trainer']['epochs'] = args.epochs
    conf['trainer']['ds_weight'] = args.ds_weight

    foldDir = os.path.join(dataDir, 'model', 'fold' + str(foldInd))
    if not os.path.exists(foldDir):
        os.makedirs(foldDir)

    if args.loss == 'FPFNLoss':
        return_weight = True
        if 'gcn' in args.model:
            if args.gcn_mode == 2:
                identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_ds_weight_' + str(args.ds_weight) + \
                             '_loss_' + args.loss + '_lamda_' + str(
                    args.lamda) + '_' + args.optimizer + '_lr_' + str(args.learning_rate)
            else:
                identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_loss_' + args.loss + '_lamda_' + str(args.lamda)\
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
                identifier = args.model + '_gcn_mode_' + str(args.gcn_mode) + '_loss_' + args.loss + '_' + args.optimizer + \
                             '_lr_' + str(args.learning_rate)
        else:
            identifier = args.model + '_loss_' + args.loss + '_' + args.optimizer + '_lr_' + str(args.learning_rate)

    if args.seed != 0:
        identifier = identifier + '_seed_' + str(args.seed)

    if args.resume:
        identifier = identifier + '_resume'
        conf['trainer']['resume'] = os.path.join(foldDir, identifier,'best_checkpoint.pytorch')
        if not os.path.exists(os.path.join(foldDir, identifier)):
            if args.seed == 0:
                src = os.path.join(foldDir, 'DeepLabv3_plus_skipconnection_3d' +
                                   '_loss_' + args.loss + '_' + args.optimizer + '_lr_' + str(args.learning_rate))
            else:
                src = os.path.join(foldDir, 'DeepLabv3_plus_skipconnection_3d' +
                                   '_loss_' + args.loss + '_' + args.optimizer + '_lr_' + str(args.learning_rate)
                                   + '_seed_' + str(args.seed))
            shutil.copytree(src=src, dst=os.path.join(foldDir, identifier))
    elif args.pre_trained:
        identifier = identifier + '_pretrained'
        conf['trainer']['pre_trained'] = os.path.join(foldDir, identifier, 'best_checkpoint.pytorch')
        if not os.path.exists(os.path.join(foldDir, identifier)):
            if args.seed == 0:
                # src = os.path.join(foldDir, 'DeepLabv3_plus_skipconnection_3d' +
                #                    '_loss_' + args.loss + '_' + args.optimizer + '_lr_' + str(args.learning_rate))

                src = os.path.join(foldDir, 'DeepLabv3_plus_skipconnection_3d' +
                                   '_loss_' + args.loss + '_' + args.optimizer + '_lr_0.001')
            else:
                # src = os.path.join(foldDir, 'DeepLabv3_plus_skipconnection_3d' +
                #                    '_loss_' + args.loss + '_' + args.optimizer + '_lr_' + str(args.learning_rate)
                #                    + '_seed_' + str(args.seed))

                src = os.path.join(foldDir, 'DeepLabv3_plus_skipconnection_3d' +
                                   '_loss_' + args.loss + '_' + args.optimizer + '_lr_0.001'
                                   + '_seed_' + str(args.seed))
            shutil.copytree(src=src, dst=os.path.join(foldDir, identifier))


    checkpoint_dir = os.path.join(foldDir, identifier)

    conf['trainer']['checkpoint_dir'] = checkpoint_dir


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
        model = DeepLabv3_plus_skipconnection_3d(nInputChannels=conf['model']['in_channels'],
                                                 n_classes=conf['model']['out_channels'],
                                                 os=16, pretrained=False, _print=True,
                                                 final_sigmoid=conf['model']['final_sigmoid'],
                                                 normalization='bn',
                                                 num_groups=8, devices=conf['device'])
    elif args.model == 'DeepLabv3_plus_gcn_skipconnection_3d':
        model = DeepLabv3_plus_gcn_skipconnection_3d(nInputChannels=conf['model']['in_channels'], n_classes=conf['model']['out_channels'],
                                  os=16, pretrained=False, _print=True, final_sigmoid=conf['model']['final_sigmoid'],
                                hidden_layers=conf['model']['hidden_layers'], devices=conf['device'],
                                                     gcn_mode=conf['model']['gcn_mode'])
    elif args.model == 'UNet3D':
        model = UNet3D(in_channels=conf['model']['in_channels'], out_channels=conf['model']['out_channels'],
                       final_sigmoid=conf['model']['final_sigmoid'], f_maps=32, layer_order='cbr')
    elif args.model == 'ResidualUNet3D':
        model = ResidualUNet3D(in_channels=conf['model']['in_channels'], out_channels=conf['model']['out_channels'],
                               final_sigmoid=conf['model']['final_sigmoid'], f_maps=32, conv_layer_order='cbr')
    # put the model on GPUs
    logger.info(f"Sending the model to '{conf['device']}'")
    # model = model.to(conf['device'])
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(conf)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(conf)

    # Create data loaders

    train_data_loader, val_data_loader, test_data_loader, f = load_data(filePath=filePath, return_weight=return_weight,
                                                                        transformer_config=conf['transformer'],
                                                                        batch_size=batch_size)
    loaders = {'train': train_data_loader, 'val': val_data_loader}

    # Create the optimizer
    optimizer = _create_optimizer(conf, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(conf, optimizer)

    # Create model trainer
    trainer = _create_trainer(conf, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders,
                              logger=logger)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
