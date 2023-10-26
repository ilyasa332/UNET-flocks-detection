import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch.birds_dataset import BirdsDataset
from pytorch.birds_utils import fix_random_seed
from pytorch.train import AutoEncoderModule, get_data
from pytorch.unet.unet_model import UNet

fix_random_seed(123)

data_github_path = '/mnt/storage/bird_data/image_data_and_labels/Data_github'
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
    test_files = test_files[:debug_size]

test_ds = BirdsDataset(test_files, num_past=num_past, diff_minutes=minutes, box=box, size=sz)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=0, shuffle=False)

device = "cuda"
model = AutoEncoderModule.load_from_checkpoint(
    "/home/theator/ziv/code/birds/pytorch/logs/lr=0.001;threshold=0.5;unet_github/version_3/checkpoints/epoch=6-step=1372.ckpt",
    model=UNet(n_channels=9, n_classes=1), loss_fn=nn.BCEWithLogitsLoss())
model.eval()

labels = []
predictions = []
for x, y in tqdm(test_dl):
    x = x.to(device=device, dtype=torch.float32)
    pred = model.model(x)
    predictions.append(F.sigmoid(pred).detach().cpu())
    labels.append(y)

predictions = torch.concatenate(predictions, dim=0)
labels = torch.concatenate(labels, dim=0)
torch.save(predictions, "preds.pkl")
torch.save(labels, "labels.pkl")
