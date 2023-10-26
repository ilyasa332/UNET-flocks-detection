import glob
from datetime import datetime
import re
import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from tqdm import tqdm


def load_files():
    # test_files_n = glob.glob('train_test/Data_github/test/Meron_flocks/2018/08/29.08.2018/*.tiff')
    # test_files_n = glob.glob('train_test/Data_github/test/Ramon_flocks/*/*/*/*.tiff')
    # test_files_n = glob.glob('train_test/Data_github/test/*_flocks/*/*/*/*.tiff')
    # test_files_n_ramon_autumn = glob.glob(r'C:\Users\asdds\Documents\ilya\work\Knafei_Silon\UNET-flocks-detection-main\train_test\Data_github/test/Ramon_flocks/*/*/*/*_VRADH.tiff')
    test_files_n_ramon_autumn = glob.glob(r'/mnt/storage/bird_data/image_data_and_labels/Data_github/test/Ramon_flocks/*/*/*/*_VRADH.tiff')
    # test_files_n_meron_autumn = glob.glob(r'C:\Users\asdds\Documents\ilya\work\Knafei_Silon\UNET-flocks-detection-main\train_test\Data_github/test/meron_flocks/*/*/*/*_VRADH.tiff')
    # test_files_n_dagan_autumn = glob.glob(r"C:\Users\asdds\Documents\ilya\work\Knafei_Silon\UNET-flocks-detection-main\train_test\new data\*\flocks\DAGAN\2018\09\08.09.2018\*.tiff")
    # test_files_n_ramon_spring = glob.glob(r"C:\Users\asdds\Documents\ilya\work\Knafei_Silon\UNET-flocks-detection-main\train_test\new data\*\flocks\RAMON\2018\04\03.04.2018\*.tiff")
    test_files_ramon_autumn = sorted(test_files_n_ramon_autumn,key=lambda file: datetime.strptime(re.search('--(.*)_', os.path.basename(file)).group(1).replace('-', ' '), '%Y%m%d %H%M%S'))
    # test_files_meron_autumn = sorted(test_files_n_meron_autumn,key=lambda file: datetime.strptime(re.search('--(.*)_', os.path.basename(file)).group(1).replace('-', ' '), '%Y%m%d %H%M%S'))
    # test_files_dagan_autumn = sorted(test_files_n_dagan_autumn,key=lambda file: datetime.strptime(re.search('--(.*)_', os.path.basename(file)).group(1).replace('-', ' '), '%Y%m%d %H%M%S'))
    # test_files_ramon_spring = sorted(test_files_n_ramon_spring,key=lambda file: datetime.strptime(re.search('--(.*)_', os.path.basename(file)).group(1).replace('-', ' '), '%Y%m%d %H%M%S'))
    # test_files = test_files_ramon
    # test_files_n = glob.glob('train_test/Spring_run/Spring_run/*/*/*/*.tiff')
    # test_files = sorted(test_files_n,key=lambda file: datetime.strptime(re.search('--(.*)_', os.path.basename(file)).group(1).replace('-', ' '), '%Y%m%d %H%M%S'))
    # print("length of ramon test files",len(test_files_ramon))
    # print("length of meron test files",len(test_files_meron))
    # return test_files
    return (test_files_ramon_autumn,)


def create_early_image_2(files, file, minuts=7,sz=(256,256),box=(29, 29, 450, 450)):
    """
    :param files: list of order images
    :param file: we check for previous images for this specific file
    :param minuts: we will add the previous images only if the gap of time between each two images
                   is less than the minuts parametr

    :return: the image concatenate to the previous images we found
    """
    # current scan
    time = datetime.strptime(re.search('--(.*)_', os.path.basename(file)).group(1).replace('-', ' '), '%Y%m%d %H%M%S')
    index = files.index(file)
    
    
    # first previous scan
    time_prev = datetime.strptime(re.search('--(.*)_', os.path.basename(files[index - 1])).group(1).replace('-', ' '), '%Y%m%d %H%M%S')
    file_prev = files[index - 1]
    delta = (time - time_prev).seconds / 60
    if delta < minuts:  # make sure the previous scan didn't happen too long ago
        image_prev = Image.open(file_prev)
        image_prev = image_prev.crop(box)
        image_prev = image_prev.resize(sz)
        image_prev = np.array(image_prev)
        
        
        # second previous scan
        time_prev_2 = datetime.strptime(re.search('--(.*)_', os.path.basename(files[index - 2])).group(1).replace('-', ' '), '%Y%m%d %H%M%S')
        file_prev_2 = files[index - 2]
        delta_2 = (time_prev - time_prev_2).seconds / 60
        if delta_2 < minuts:
            image_prev_2 = Image.open(file_prev_2)
            image_prev_2 = image_prev_2.crop(box)
            image_prev_2 = image_prev_2.resize(sz)
            image_prev_2 = np.array(image_prev_2)
            
            # stack two previous images
            image_prev_all = np.concatenate((image_prev, image_prev_2), axis=2)
        else:
            image_prev_all = []

            
    else:
        image_prev_all = []

    return image_prev_all


def Preprocessor(files, sz=(256,256),box=(29, 29, 450, 450)):  # returns a tensor with a ready input for the model from the image files
    images = []
    test_y = []
    indices = []
    for i, f in tqdm(enumerate(files)):
        # open and preprocess the current raw image
        raw = Image.open(f)
        raw = raw.crop(box)
        raw = raw.resize(sz)
        img = np.array(raw)
        
        # create and stack previous 2 scans for each image. if there are less than 2 scans available, discard the image.
        prev_img = create_early_image_2(files, f)
        if len(prev_img) == 0:
            continue
        else:
            img = np.concatenate((img, prev_img), axis=2)
        
        # stack all 3 images into the tensor
        images.append(img)
        
        
        
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
            mask = Image.open(mask_path)
            mask = np.asarray(mask)
            if mask.shape == (480, 480, 2):
                tempmask = mask[:,:,1].copy()
                tempmask[mask[:,:,1] != 0] = 0
                tempmask[mask[:,:,1] == 0] = 255
                tempmask = tempmask + mask[:,:,0]
                tempmask[tempmask != 0] = 255
                mask = tempmask
            mask = Image.fromarray(mask)
            mask = mask.crop(box)
            mask = np.array(mask.resize(sz))

            mask[mask != 0] = 255

        test_y.append(mask)
        
        indices.append(i)
        
    return np.array(images) / 255, np.array(test_y), indices