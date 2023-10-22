import glob
import os
import re
from datetime import datetime

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from pytorch.birds_dataset import BirdsDataset
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from pytorch.birds_utils import fix_random_seed
from pytorch.model import UNet

fix_random_seed(123)


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.train_auc = BinaryAUROC()
        self.valid_auc = BinaryAUROC()

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        x, y = batch
        output = self.model(x.to(torch.float32))
        loss = self.loss_fn(output, y.to(torch.float32))
        self.train_auc.update(output.detach().cpu(), y.cpu())
        self.log("loss_train", loss.item(), on_step=True)
        return loss

    def on_train_epoch_end(self):
        # pass
        self.log('auc_train', self.train_auc.compute())

    def on_train_epoch_start(self) -> None:
        del self.train_auc
        del self.valid_auc
        self.train_auc = BinaryAUROC()
        self.valid_auc = BinaryAUROC()
        # pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(self.model.parameters(), lr=0.01)

    def validation_step(self, batch, batch_index) -> STEP_OUTPUT:
        x, y = batch
        output = self.model(x.to(torch.float32))
        loss = self.loss_fn(output, y.to(torch.float32))
        self.valid_auc.update(output.detach().cpu(), y.cpu())
        self.log("loss_valid", loss.item(), on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('auc_valid', self.valid_auc.compute())
        pass


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

debug = False
debug_size = 500
box = (29, 29, 450, 450)
num_past = 2
minutes = 7
sz = (256, 256)

if debug:
    print(f"DEBUG MODE")
    train_files = train_files[:debug_size]
    test_files = test_files[:debug_size]

train_ds = BirdsDataset(train_files, num_past=num_past, diff_minutes=minutes, box=box, size=sz)
train_dl = DataLoader(train_ds, batch_size=32, num_workers=12, shuffle=True)

test_ds = BirdsDataset(test_files, num_past=num_past, diff_minutes=minutes, box=box, size=sz)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=12, shuffle=False)

device = "cuda"

lightning_model = AutoEncoderModule(model=UNet(), loss_fn=nn.BCELoss())
logger = TensorBoardLogger(save_dir='logs')

trainer = pl.Trainer(max_epochs=30, devices='auto', accelerator='gpu', logger=logger,
                     log_every_n_steps=1)
trainer.fit(model=lightning_model, train_dataloaders=train_dl, val_dataloaders=test_dl)
