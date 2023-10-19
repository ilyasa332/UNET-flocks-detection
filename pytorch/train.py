from datetime import datetime
import glob
import os
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from experiments.birds.dataset import image_generator
from experiments.birds.model import UNet


def get_data(src_dir: str, phase: str):
    tiff_path = os.path.join(src_dir, f'{phase}/*/*/*/*/*VRADH.tiff')
    files_n = glob.glob(tiff_path)

    # sort the file by the data and time
    files = sorted(files_n, key=lambda file: datetime.strptime(
        re.search('--(.*)_VRADH', os.path.basename(file)).group(1).replace('-', ' '), '%Y%m%d %H%M%S'))

    print(f"length of {phase} files: {len(files)}")
    return files


data_github_path = '/mnt/storage/bird_data/image_data_and_labels/Data_github'
train_files = get_data(data_github_path, "train")
test_files = get_data(data_github_path, "test")

box = (29, 29, 450, 450)
num_past = 2
minutes = 7
batch_size = 128
sz = (256, 256)


class IterDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


train_generator = image_generator(train_files, num_past=num_past, minutes=minutes, box=box, batch_size=4)
train_dl = DataLoader(IterDataset(train_generator), batch_size=1)

# valid_generator = image_generator(test_files, num_past=num_past, minutes=minutes, box=box, batch_size=1)
# valid_dl = DataLoader(valid_generator, batch_size=1)

device = "cuda"

model = UNet()
model.to(device)
loss_fn = nn.BCELoss()

optimizer = torch.optim.RMSprop(model.parameters())


for x, y in train_dl:
    x = x.to(device)
    y = y.to(device)
    output = model(x.permute((0, 1, 4, 2, 3)).squeeze(0))
    loss = loss_fn(output.squeeze(), y.squeeze(0, -1).to(torch.float32))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)
