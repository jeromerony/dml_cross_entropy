from PIL import Image
from torch.utils.data import Dataset

from utils.data.utils import load_data


class ImageDataset(Dataset):
    def __init__(self, samples: list, transform, preload: bool = False, num_workers=None):
        self.transform = transform
        self.samples = samples
        self.targets = [label for _, label in self.samples]

        self.preloaded = False
        if preload:
            image_paths = [image_path for image_path, _ in self.samples]
            self.images = load_data(image_paths, num_workers=num_workers)
            self.preloaded = True
            print(self.__class__.__name__ + ' loaded with {} images'.format(len(self.images.keys())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        if self.preloaded:
            image = self.images[image_path].convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label, index
