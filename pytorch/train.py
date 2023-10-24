import glob
import os
import re
from datetime import datetime

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from pytorch.birds_dataset import BirdsDataset, DummyDataset
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from pytorch.birds_utils import fix_random_seed
from pytorch.unet.unet_model import UNet
from pytorch.visualize import visualize_predictions

fix_random_seed(123)


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fn):
        super().__init__()
        self.tr = 0.013
        self.model = model
        self.loss_fn = loss_fn
        self.train_auc = BinaryAUROC()
        self.valid_auc = BinaryAUROC()

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        x, y = batch
        output = self.model(x.to(torch.float32))
        loss = self.loss_fn(output, y.to(torch.float32))
        # loss = ((loss * y) / self.tr + (loss * (1 - y)) / (1 - self.tr)).mean()
        self.train_auc.update(F.sigmoid(output).detach().cpu(), y.cpu())
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def validation_step(self, batch, batch_index) -> STEP_OUTPUT:
        x, y = batch
        output = self.model(x.to(torch.float32))
        loss = self.loss_fn(output, y.to(torch.float32))
        # loss = ((loss * y) / self.tr + (loss * (1 - y)) / (1 - self.tr)).mean()
        self.valid_auc.update(F.sigmoid(output).detach().cpu(), y.cpu())
        self.log("loss_valid", loss.item(), on_epoch=True)
        visualize_predictions(x, F.sigmoid(output), y, self.logger, step=self.trainer.current_epoch, batch_index=batch_index,
                              threshold=threshold)
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

lr = 0.001
threshold = 0.5
num_epochs = 20

if debug:
    print(f"DEBUG MODE")
    train_files = train_files[:debug_size]
    test_files = test_files[:debug_size]

train_ds = BirdsDataset(train_files, num_past=num_past, diff_minutes=minutes, box=box, size=sz)
# train_ds = DummyDataset(train_ds, indices=(352, 353))
train_dl = DataLoader(train_ds, batch_size=32, num_workers=12, shuffle=True)

test_ds = BirdsDataset(test_files, num_past=num_past, diff_minutes=minutes, box=box, size=sz)
# test_ds = DummyDataset(train_ds, indices=(352, 353), trim_len=True)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=12, shuffle=False)

device = "cuda"
lightning_model = AutoEncoderModule(model=UNet(n_channels=9, n_classes=1), loss_fn=nn.BCEWithLogitsLoss())
logger = TensorBoardLogger(save_dir='logs', name=f"lr={lr};threshold={threshold};unet_github")

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(max_epochs=num_epochs, devices=[0], accelerator='gpu', logger=logger,
                     log_every_n_steps=1, callbacks=[lr_monitor], max_steps=num_epochs * len(train_dl))
trainer.fit(model=lightning_model, train_dataloaders=train_dl, val_dataloaders=test_dl)
