import math
import os
from copy import deepcopy
from functools import partial
from pprint import pprint

import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler
from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import state_dict_to_cpu, SmoothCrossEntropy
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train, evaluate

ex = sacred.Experiment('Metric Learning', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    epochs = 20
    lr = 0.02
    momentum = 0.
    nesterov = False
    weight_decay = 5e-4
    scheduler = 'warmcos'

    visdom_port = None
    visdom_freq = 20
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = os.path.join('results', 'temp')

    no_bias_decay = True
    label_smoothing = 0.1
    temperature = 1.


@ex.capture
def get_optimizer_scheduler(parameters, loader_length, epochs, lr, momentum, nesterov, weight_decay, scheduler,
                            lr_step=None):
    optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                    nesterov=True if nesterov and momentum else False)
    if epochs == 0:
        scheduler = None
    elif scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * loader_length, eta_min=0)
    elif scheduler == 'warmcos':
        warm_cosine = lambda i: min((i + 1) / 100, (1 + math.cos(math.pi * i / (epochs * loader_length))) / 2)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_cosine)
    elif scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, lr_step * loader_length)
    elif scheduler == 'warmstep':
        warm_step = lambda i: min((i + 1) / 100, 1) * 0.1 ** (i // (lr_step * loader_length))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_step)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, epochs * loader_length)
    return optimizer, scheduler


@ex.automain
def main(epochs, cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, no_bias_decay, label_smoothing,
         temperature):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model = get_model(num_classes=loaders.num_classes)
    class_loss = SmoothCrossEntropy(epsilon=label_smoothing, temperature=temperature)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(parameters=parameters, loader_length=len(loaders.train))

    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, recall=recall_ks, query_loader=loaders.query,
                            gallery_loader=loaders.gallery)

    # setup best validation logger
    metrics = eval_function()
    if callback is not None:
        callback.scalars(['l2', 'cosine'], 0, [metrics.recall['l2'][1], metrics.recall['cosine'][1]],
                         title='Val Recall@1')
    pprint(metrics.recall)
    best_val = (0, metrics.recall, deepcopy(model.state_dict()))

    torch.manual_seed(seed)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        train(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer,
              scheduler=scheduler, epoch=epoch, callback=callback, freq=visdom_freq, ex=ex)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        metrics = eval_function()
        print('Validation [{:03d}]'.format(epoch)), pprint(metrics.recall)
        ex.log_scalar('val.recall_l2@1', metrics.recall['l2'][1], step=epoch + 1)
        ex.log_scalar('val.recall_cosine@1', metrics.recall['cosine'][1], step=epoch + 1)

        if callback is not None:
            callback.scalars(['l2', 'cosine'], epoch + 1,
                             [metrics.recall['l2'][1], metrics.recall['cosine'][1]], title='Val Recall')

        # save model dict if the chosen validation metric is better
        if metrics.recall['cosine'][1] >= best_val[1]['cosine'][1]:
            best_val = (epoch + 1, metrics.recall, deepcopy(model.state_dict()))

    # logging
    ex.info['recall'] = best_val[1]

    # saving
    save_name = os.path.join(temp_dir, '{}_{}.pt'.format(ex.current_run.config['model']['arch'],
                                                         ex.current_run.config['dataset']['name']))
    torch.save(state_dict_to_cpu(best_val[2]), save_name)
    ex.add_artifact(save_name)

    if callback is not None:
        save_name = os.path.join(temp_dir, 'visdom_data.pt')
        callback.save(save_name)
        ex.add_artifact(save_name)

    return best_val[1]['cosine'][1]
