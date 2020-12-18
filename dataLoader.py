import numpy as np
from torch.utils.data import Dataset, sampler
from torchvision import transforms
from PIL import Image
import glob

# For VGG
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        super(ImageDataset, self).__init__()
        # root directory
        self.root_dir = root_dir
        # Get a list of image names in file directory
        self.images = glob.glob(self.root_dir+'**/*.*')
        self.transform = transform

    def __getitem__(self, index):
        """
        Load an input image at this index from the root, convert it to the format VGG
        accepts.

        return: img tensor
        """

        img_name = self.images[index]
        img = Image.open(str(img_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'Image Dataset'


# transform for the training set
def train_transform(size, crop_size):

    transform = transforms.Compose([
                transforms.Resize(size),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                _NORMALIZE
            ])
    return transform


# transform for the testing set
def test_transform(size=0, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(_NORMALIZE)
    return transforms.Compose(transform_list)


# Infinite iterator
def InfiniteSamplerIterator(n):

    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Infinite random sampler which implement InfiniteSamplerIterator
class InfiniteSampler(sampler.Sampler):

    def __init__(self, num):
        self.num_samples = num

    def __iter__(self):
        return iter(InfiniteSamplerIterator(self.num_samples))

    def __len__(self):
        return 2 ** 31
