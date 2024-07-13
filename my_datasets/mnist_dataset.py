import torch
from torchvision import datasets
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, CenterCrop, Normalize
from torch.utils.data import DataLoader


def mnist_t(img_size, mean, std):
    t = Compose([
        Resize(img_size),
        RandomCrop((img_size, img_size)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return t

def collat_fn(batch):
    imgs = [sample[0] for sample in batch]
    return torch.stack(imgs)


def get_mnist_train_data(data_config):
    transform = mnist_t(**data_config.transform_config)
    dataset = datasets.MNIST(**data_config.dataset_config, train=True, download=False, transform=transform)
    data_loader = DataLoader(dataset=dataset, shuffle=True, collate_fn=collat_fn, **data_config.data_loader_config, )
    return dataset, data_loader



def get_mnist_test_data(data_config):
    transform = mnist_t(**data_config.transform_config)
    dataset = datasets.MNIST(**data_config.dataset_config, train=False, download=False, transform=transform)
    data_loader = DataLoader(dataset=dataset, shuffle=False, collate_fn=collat_fn, **data_config.data_loader_config, )
    return dataset, data_loader
