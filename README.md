# Automatic detection of soaring bird migration using weather radars by deep learning method

This repository contains UNET network for automatically detecting soaring birds by weather radar, developed by Schekler et al. at the University of Haifa, Israel.

For details, see our publication:



## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in colab. following Environement and Library needed to run the code:

- Python 3
- Keras - tensorflow backend version 1.x

## Data 
We could only share the PPI images created from the original h5 radar files (and not the h5 files).
For running a demo with our data (radial velocity images) follow the bellow steps:

### Run Demo
1- Download the train and test data from [this link](https://drive.google.com/file/d/1Hrb1F7lzfVPqyzXJq-WQ46zSClk7ks0F/view?usp=sharing) 
2- Open 'UNET_Soaring birds_model_f.ipynb' 
3- Copy the zip folder of the data to the colab files area and run the code training the UNET model with our data
4- Run 'evaluate_performance_f.ipynb' for performance evalution with the best epoch from the previous code


For prediciting your data with our trainned model,follow the bellow steps:

1- Create from your h5 radar files PPI images with 'creating_ppi.R' (in the 'prepare_data' folder). The model uses 2 previous images of each image we want to predict so you have to have consecutive images.
2- Download our best epoch from [this link](https://drive.google.com/file/d/1hnWelWk0rSyUfAXgGJMQa_PCyip97_sc/view?usp=sharing)
3- Run 'evaluate_performance_f.ipynb'


In case you want to add more data for the training follow the bellow steps:

1- Create PPI images with 'creating_ppi.R' in the 'prepare_data' folder.
2- You need to tag the images. We used labal- studio https://labelstud.io/
3- In case you did used label-studio, the program creates a few images if you tag the same image a few times (for adding/ correcting previous tag) and in addition, the mask name do not get the name of the origin image. The code prepare_img_mask.py in the 'prepare_data' folder concatanate masks of the same image and in addition, conact the image and mask names by their file names. 
