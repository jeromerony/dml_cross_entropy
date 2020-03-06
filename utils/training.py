from functools import partial
from typing import NamedTuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Experiment
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom_logger import VisdomLogger

from utils.metrics import AverageMeter, recall_at_ks


def train(model: nn.Module,
          loader: DataLoader,
          class_loss: nn.Module,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          epoch: int,
          callback: VisdomLogger,
          freq: int,
          ex: Experiment = None) -> None:
    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))
        logits, features = model(batch)
        loss = class_loss(logits, labels).mean()
        acc = (logits.detach().argmax(1) == labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(acc)

        if callback is not None and not (i + 1) % freq:
            step = epoch + i / loader_length
            callback.scalar('xent', step, train_losses.last_avg, title='Train Losses')
            callback.scalar('train_acc', step, train_accs.last_avg, title='Train Acc')

    if ex is not None:
        for i, (loss, acc) in enumerate(zip(train_losses.values_list, train_accs.values_list)):
            step = epoch + i / loader_length
            ex.log_scalar('train.loss', loss, step=step)
            ex.log_scalar('train.acc', acc, step=step)


class _Metrics(NamedTuple):
    loss: float
    accuracy: float
    recall: Dict[str, Dict[int, float]]


def evaluate(model: nn.Module,
             query_loader: DataLoader,
             gallery_loader: Optional[DataLoader] = None,
             xent: bool = False,
             recall: Optional[List[int]] = None) -> _Metrics:
    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    all_query_labels = []
    all_query_features = []
    all_gallery_features = None
    all_gallery_labels = None
    xent_losses = []
    all_predictions = []

    with torch.no_grad():
        for batch, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, labels = map(to_device, (batch, labels))
            logits, features = model(batch)

            all_query_labels.append(labels)
            if recall is not None:
                all_query_features.append(features)
            if xent:
                xent_losses.append(F.cross_entropy(logits, labels, reduction='none'))
            all_predictions.append(logits.argmax(1))

        if gallery_loader is not None and recall is not None:
            all_gallery_features = []
            all_gallery_labels = []
            for batch, labels, _ in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
                batch, labels = map(to_device, (batch, labels))
                features = model(batch)[1]

                all_gallery_labels.append(labels)
                all_gallery_features.append(features)

            all_gallery_labels = torch.cat(all_gallery_labels, 0)
            all_gallery_features = torch.cat(all_gallery_features, 0)
        torch.cuda.empty_cache()

        all_query_labels = torch.cat(all_query_labels, 0)
        if recall is not None:
            all_query_features = torch.cat(all_query_features, 0)
            recall_function = partial(
                recall_at_ks, query_features=all_query_features, query_labels=all_query_labels, ks=recall,
                gallery_features=all_gallery_features, gallery_labels=all_gallery_labels
            )
            recalls = {'l2': recall_function(), 'cosine': recall_function(cosine=True)}
        if xent:
            xent_losses = torch.cat(xent_losses, 0)
            loss = xent_losses.mean().item()
        all_predictions = torch.cat(all_predictions, 0)
        acc = (all_predictions == all_query_labels).float().mean().item()

    return _Metrics(loss=loss if xent else None, accuracy=acc, recall=recalls if recall is not None else None)
