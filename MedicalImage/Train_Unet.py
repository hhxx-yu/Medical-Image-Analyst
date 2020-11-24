#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:45:59 2020

@author: ryan
"""
#%%[]
import os
os.chdir(os.path.dirname(__file__))
import numpy as np
from data_generate import DataGen
import random
import matplotlib.pyplot as plt
from Unet import UNet
from glob import glob
#%%
#setting image input parameters
image_size = 256
train_path = "Preprocessed_Data/Train"
valid_path="Preprocessed_Data/Valid"
epochs = 15
batch_size = 8
#%%
#create generator objects for training
#I don't know how to use augmentated image in a customize generator, so i just put them all in training set
def get_img_name(file):
    return file.split("/")[-1][:-4]
train_ids=glob("Preprocessed_Data/Train/image/*.tif")
aug_ids=glob("Preprocessed_Data/Aug/image_aug/*.tif")
train_ids=list(map(get_img_name,train_ids))
aug_ids=list(map(get_img_name,aug_ids))
train_ids=train_ids+aug_ids
valid_ids=sorted(glob("Preprocessed_Data/Valid/image/*.tif"))
valid_ids=list(map(get_img_name,valid_ids))
valid_gen = DataGen(valid_ids,path='Preprocessed_Data/Valid',image_size=image_size,shuffle=False)

train_gen = DataGen(train_ids,path='Preprocessed_Data/Train',image_size=image_size)
'''
tests for Data generator
x,y=train_gen.__getitem__(440)
for i in range(0,8):
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(np.reshape(y[i]*255, (image_size, image_size)), cmap="gray")
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(x[i,:,:,1])
    '''
#%%
#train model
train_steps = len(train_gen)
valid_steps = len(valid_gen)
model_aug=UNet()
model_aug.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model_aug.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
checkpoint = ModelCheckpoint("final_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True)
callbacks_list = [EarlyStopping(monitor='val_loss', patience=3),checkpoint]
#fit model
history=model_aug.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=15,callbacks=callbacks_list)
#model_aug.save("model_aug_fine.h5")
model_aug.save("final_model_416.h5")
#%%
#use other loss function like dice loss, currently not working, trying to solve that.
'''
from keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
#try dice loss
#model_aug_dice=UNet()
#model_aug_dice.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
#model_aug_dice.summary()
#history_d=model_aug_dice.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=12)
    '''
