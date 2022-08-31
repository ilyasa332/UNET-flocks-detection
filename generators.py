import numpy as np
from PIL import Image
import os

from create_previous_images import create_early_image_2



box = (29, 29, 450, 450)
num_past = 2
minuts = 7


def image_generator(files, batch_size=32, sz=(256, 256)):
    while True:

        # extract a random batch
        batch = np.random.choice(files, size=batch_size)

        # variables for collecting batches of inputs and outputs
        batch_x = []
        batch_y = []

        for f in batch:

            # preprocess the raw images
            raw = Image.open(f)
            raw = raw.crop(box)
            raw = raw.resize(sz)
            raw = np.array(raw)

            prev_image = create_early_image_2(files, f, num_past, minuts)
            if len(prev_image) == 0:

                continue
            else:
                raw_prev = np.concatenate((raw, prev_image), axis=2)

            batch_x.append(raw_prev)

            # get the masks. Note that masks are png files
            f_path = os.path.dirname(f)
            num_f = str(int(os.path.basename(f).split('-')[0]))
            mask_file = [i for i in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, i)) and
                         num_f == (os.path.basename(i).split('-')[0]) and '.png' in i]
            if len(mask_file) == 0:
                mask = np.zeros((256, 256))
            else:
                mask_file = str(mask_file)[2:-2]
                mask_path = os.path.join(f_path, mask_file)
                mask = Image.open(mask_path).convert('L')
                mask = mask.crop(box)
                mask = np.array(mask.resize(sz))

                mask[mask != 0] = 1

            batch_y.append(mask)

        # preprocess a batch of images and masks
        batch_x = np.array(batch_x) / 255.
        batch_y = np.array(batch_y)

        batch_y = np.expand_dims(batch_y, 3)

        yield (batch_x, batch_y)

