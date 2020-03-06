from typing import Dict, List, Optional

import faiss
import torch
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value on device"""

    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg


@torch.no_grad()
def recall_at_ks(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = False) -> Dict[int, float]:
    """
    Compute the recall between samples at each k. This function uses about 8GB of memory.

    Parameters
    ----------
    query_features : torch.Tensor
        Features for each query sample. shape: (num_queries, num_features)
    query_labels : torch.LongTensor
        Labels corresponding to the query features. shape: (num_queries,)
    ks : List[int]
        Values at which to compute the recall.
    gallery_features : torch.Tensor
        Features for each gallery sample. shape: (num_queries, num_features)
    gallery_labels : torch.LongTensor
        Labels corresponding to the gallery features. shape: (num_queries,)
    cosine : bool
        Use cosine distance between samples instead of euclidean distance.

    Returns
    -------
    recalls : Dict[int, float]
        Values of the recall at each k.

    """
    offset = 0
    if gallery_features is None and gallery_labels is None:
        offset = 1
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    if cosine:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    max_k = max(ks)
    index_function = faiss.GpuIndexFlatIP if cosine else faiss.GpuIndexFlatL2
    index = index_function(res, g_f.shape[1], flat_config)
    index.add(g_f)
    closest_indices = index.search(q_f, max_k + offset)[1]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (q_l[:, None] == g_l[indices]).any(1).mean()
    return {k: round(v * 100, 2) for k, v in recalls.items()}
