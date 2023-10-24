import cv2
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger


def visualize_predictions(inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor, logger: TensorBoardLogger,
                          batch_index: int, max_imgs: int = 32, step: int = None, threshold: int = 0.5):
    for i, (x, output, y) in enumerate(zip(inputs, outputs, labels)):
        if y.sum().item() == 0:
            continue
        x_img = x[:3].detach().cpu().numpy().transpose(1, 2, 0)
        y_img = cv2.cvtColor((y[0].detach().cpu().numpy()), cv2.COLOR_GRAY2RGB)
        output_img = cv2.cvtColor(output[0].detach().cpu().numpy(), cv2.COLOR_GRAY2RGB)
        pred_img = cv2.cvtColor((output[0] > threshold).to(output.dtype).detach().cpu().numpy(), cv2.COLOR_GRAY2RGB)
        img = np.concatenate((x_img, y_img, output_img, pred_img), axis=1)
        logger.experiment.add_image(f"vis-{i + batch_index * len(x)}", img.transpose(2, 0, 1), global_step=step)
