import multiprocessing as mp
from time import time

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

if __name__ == '__main__':
    train_reader = MNIST("data/temp", train=True,
                         transform=transforms.Compose([
                             transforms.Resize(128),
                             transforms.ToTensor(),
                             transforms.Normalize((0.7,), (0.7,))
                         ]))

    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = DataLoader(train_reader, shuffle=True, num_workers=num_workers, batch_size=10, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
