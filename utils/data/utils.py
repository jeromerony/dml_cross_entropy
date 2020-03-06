from functools import partial
from multiprocessing.pool import Pool
from random import sample
from typing import Union, List

import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision.transforms import functional as F


class RandomReplacedIdentitySampler(Sampler):
    def __init__(self, labels: Union[List[int], torch.Tensor], batch_size: int, num_identities: int, num_iterations: int):
        self.num_identities = num_identities
        self.num_iterations = num_iterations
        self.samples_per_id = batch_size // num_identities
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.counts = torch.bincount(self.labels)
        self.label_indices = [torch.nonzero(self.labels == i).squeeze(1).tolist() for i in range(len(self.counts))]

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        possible_ids = [i for i in range(len(self.label_indices)) if len(self.label_indices[i]) >= self.samples_per_id]

        for i in range(self.num_iterations):
            batch = []
            selected_ids = sample(possible_ids, k=self.num_identities)
            for s_id in selected_ids:
                batch.extend(sample(self.label_indices[s_id], k=self.samples_per_id))
            yield batch


def load_image(image_path: str, resize=None, min_resize=None) -> tuple:
    image = Image.open(image_path)
    if resize is not None:
        image = image.resize(resize, resample=Image.LANCZOS)
    elif min_resize:
        image = F.resize(image, min_resize, interpolation=Image.LANCZOS)
    return image_path, image.copy()


def load_data(samples: list, resize=None, min_resize=None, num_workers: int = None) -> dict:
    load_image_partial = partial(load_image, resize=resize, min_resize=min_resize)
    if num_workers:
        with Pool(num_workers) as p:
            images = p.map(load_image, samples)
    else:
        images = map(load_image_partial, samples)
    images = dict(images)
    return images
