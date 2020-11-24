#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:15:19 2020

@author: ryan
"""

import os
import numpy as np
from data_generate import DataGen
import random
import matplotlib.pyplot as plt
from Unet import UNet
from glob import glob
from outline import *
os.chdir(os.path.dirname(__file__))
#%%
#setting image input parameters
image_size = 256
train_path = "brain_tumor_preprocessed_data/Train"
valid_path="brain_tumor_preprocessed_data/Valid"
test_path="brain_tumor_preprocessed_data/Test"
epochs = 15
batch_size = 4

def get_img_name(file):
    return file.split("/")[-1][:-4]
train_ids=glob("brain_tumor_preprocessed_data/Train/image/*.jpg")
aug_ids=glob("brain_tumor_preprocessed_data/Train/image_aug/*.jpg")
train_ids=list(map(get_img_name,train_ids))
aug_ids=list(map(get_img_name,aug_ids))
train_ids=train_ids+aug_ids
valid_ids=sorted(glob("brain_tumor_preprocessed_data/Valid/image/*.jpg"))
valid_ids=list(map(get_img_name,valid_ids))
valid_gen = DataGen(valid_ids,batch_size=batch_size,path=valid_path,image_size=image_size,form="jpg",shuffle=False)

train_gen = DataGen(train_ids,batch_size=batch_size,path=train_path,form="jpg",image_size=image_size)
'''
x,y=train_gen.__getitem__(18)
for i in range(0,batch_size):
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(np.reshape(y[i]*255, (image_size, image_size)), cmap="gray")
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(x[i,:,:,0],cmap="gray")
'''

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
model_segmentation = load_model("final_model_416.h5")
model=model_segmentation
for i,layer in enumerate(model.layers):
    if i<18:
        layer.trainable = False
model.summary()
import tensorflow as tf
import tensorflow.keras.backend as K

#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from loss_and_metrics import create_weighted_binary_crossentropy,dice_coef,dice_coef_loss

weighted_binary_crossentropy=create_weighted_binary_crossentropy(0.2, 0.8)
model.compile('adam',
                  loss=weighted_binary_crossentropy, metrics=[dice_coef,dice_coef_loss])

train_steps = len(train_gen)
valid_steps = len(valid_gen)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
checkpoint = ModelCheckpoint("final_model_tranferlearning.h5", monitor='val_dice_coef', verbose=1, save_best_only=True)
reduce_lr=ReduceLROnPlateau(monitor="dice_coef_loss",factor=0.5,patience=3,verbose=1,cooldown=0)
callbacks_list = [reduce_lr,checkpoint,EarlyStopping(monitor="dice_coef_loss", patience=10)]
#fit model
history=model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=50,callbacks=callbacks_list)
weighted_binary_crossentropy2=create_weighted_binary_crossentropy(0.3, 0.7)
model.compile('adam',
                  loss=weighted_binary_crossentropy2, metrics=[dice_coef,dice_coef_loss])
history2=model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=50,callbacks=callbacks_list)
weighted_binary_crossentropy2=create_weighted_binary_crossentropy(0.4, 0.6)
model.compile('adam',
                  loss=weighted_binary_crossentropy3, metrics=[dice_coef,dice_coef_loss])
history3=model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=50,callbacks=callbacks_list)

model.save("final_model_tumor.h5")
dice_epoch=history.history['dice_coef']+history2.history['dice_coef']+history3.history['dice_coef']
plt.plot(dice_epoch)
plt.title('Model dice coefficient per epoch')
plt.ylabel('dice coefficient')
plt.xlabel('Epoch')
plt.axvline(x=16,color="red")
plt.axvline(x=66,color="orange")
plt.text(12,0.9,"increase 0 weights")
plt.text(62,0.9,"increase 0 weights the second time")
plt.show()

