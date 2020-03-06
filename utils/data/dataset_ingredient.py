import os
from typing import NamedTuple, Optional

from sacred import Ingredient
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torchvision import transforms

from .image_dataset import ImageDataset
from .utils import RandomReplacedIdentitySampler

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def config():
    train_file = 'train.txt'
    test_file = 'test.txt'

    batch_size = 128
    test_batch_size = 256
    sampler = 'random'

    preload = False  # load all images into RAM to avoid IO
    num_workers = 8  # number of workers used ot load the data
    pin_memory = False  # use the pin_memory option of DataLoader

    crop_size = 224
    scale = (0.16, 1)
    ratio = (3. / 4., 4. / 3.)

    recalls = [1, 2, 4, 8, 16, 32]


@data_ingredient.named_config
def cub():
    name = 'cub'
    data_path = 'data/CUB_200_2011'
    resize = 256
    color_jitter = (0.25, 0.25, 0.25, 0)


@data_ingredient.named_config
def cars():
    name = 'cars'
    data_path = 'data/CARS_196'
    resize = (256, 256)
    color_jitter = (0.3, 0.3, 0.3, 0.1)
    ratio = (1, 1)


@data_ingredient.named_config
def sop():
    name = 'sop'
    data_path = 'data/Stanford_Online_Products'
    resize = (256, 256)
    recalls = [1, 10, 100, 1000]


@data_ingredient.named_config
def inshop():
    name = 'inshop'
    test_file = ('test_query.txt', 'test_gallery.txt')
    data_path = 'data/InShop'
    recalls = [1, 10, 20, 30, 40, 50]


@data_ingredient.capture
def get_transforms(crop_size, scale, ratio, resize=None, rotate=None, color_jitter=None):
    train_transform, test_transform = [], []
    if resize is not None:
        test_transform.append(transforms.Resize(resize))
        train_transform.append(transforms.Resize(resize))
    if rotate is not None:
        train_transform.append(transforms.RandomRotation(rotate))
    if color_jitter is not None:
        train_transform.append(transforms.ColorJitter(*color_jitter))
    train_transform.extend([transforms.RandomResizedCrop(size=crop_size, scale=scale, ratio=ratio),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])

    test_transform.extend([transforms.CenterCrop(size=crop_size),
                           transforms.ToTensor()])

    return transforms.Compose(train_transform), transforms.Compose(test_transform)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


@data_ingredient.capture
def get_sets(name, data_path, train_file, test_file, preload, num_workers):
    train_transform, test_transform = get_transforms()

    train_lines = read_file(os.path.join(data_path, train_file))
    train_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in train_lines]
    train_set = ImageDataset(train_samples, transform=train_transform, preload=preload, num_workers=num_workers)

    if isinstance(test_file, list) and len(test_file) == 2:
        query_lines = read_file(os.path.join(data_path, test_file[0]))
        gallery_lines = read_file(os.path.join(data_path, test_file[1]))
        query_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in query_lines]
        gallery_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in
                           gallery_lines]
        query_set = ImageDataset(query_samples, transform=test_transform, preload=preload, num_workers=num_workers)
        gallery_set = ImageDataset(gallery_samples, transform=test_transform, preload=preload, num_workers=num_workers)
    else:
        query_lines = read_file(os.path.join(data_path, test_file))
        query_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in query_lines]
        query_set = ImageDataset(query_samples, transform=test_transform, preload=preload, num_workers=num_workers)
        gallery_set = None

    return train_set, (query_set, gallery_set)


class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    gallery: Optional[DataLoader] = None


@data_ingredient.capture
def get_loaders(batch_size, test_batch_size, num_workers, pin_memory, sampler, recalls,
                num_iterations=None, num_identities=None):
    train_set, (query_set, gallery_set) = get_sets()

    if sampler == 'random':
        train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=True)
    elif sampler == 'random_id':
        train_sampler = RandomReplacedIdentitySampler(train_set.targets, batch_size, num_identities, num_iterations)
    else:
        raise ValueError('Invalid choice of sampler ({}).'.format(sampler))
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    query_loader = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = None
    if gallery_set is not None:
        gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers,
                                    pin_memory=pin_memory)

    return MetricLoaders(train=train_loader, query=query_loader, gallery=gallery_loader,
                         num_classes=max(train_set.targets) + 1), recalls
