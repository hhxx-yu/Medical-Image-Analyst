#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:43:49 2020

@author: ryan
"""
#%%
#drawing outlines for the true and predicted mask
import os
import numpy as np
from tensorflow.keras.models import load_model
from outline import *
os.chdir(os.path.dirname(__file__))
model = load_model("final_model_416.h5")
from outline import *
valid_ids=sorted(glob("Preprocessed_Data/Valid/image/*.tif"))
test_ids=sorted(glob("Preprocessed_Data/Test/image/*.tif"))
def get_img_name(file):
    return file.split("/")[-1][:-4]
valid_ids=list(map(get_img_name,valid_ids))
test_ids=list(map(get_img_name,test_ids))
valid_gen = DataGen(valid_ids,path='Preprocessed_Data/Valid',image_size=image_size,shuffle=False)
test_gen=DataGen(test_ids,path='Preprocessed_Data/Test',image_size=image_size,shuffle=False)
#save the predicted image of valid set, yellow line is the true mask, green line is the predicted mask
for i in range(len(valid_gen)):
    x,y=valid_gen.__getitem__(i)
    result = model.predict(x)
    for j in range(x.shape[0]):
        image = outline(x[j],y[j,:,:,0],color=[1,1,0])
        image = outline(image, result[j,:,:, 0], color=[0, 1, 0])
        plt.imsave("output/valid/"+valid_gen.ids[8*i+j]+"pred"+".png",image)
#save the predicted image of test set, yellow line is the true mask, green line is the predicted mask
for i in range(len(test_gen)):
    x,y=test_gen.__getitem__(i)
    result = model.predict(x)
    for j in range(x.shape[0]):
        image = outline(x[j],y[j,:,:,0],color=[1,1,0])
        image = outline(image, result[j,:,:, 0], color=[0, 1, 0])
        plt.imsave("output/test/"+test_gen.ids[8*i+j]+"pred"+".png",image)
#%%
#draw a ROC curve
result=model.predict(test_gen)
for i in range(len(test_gen)):
    if(i==0):
        true=test_gen.__getitem__(0)[1]
    else:
        true=np.append(true,test_gen.__getitem__(i)[1],axis=0)    
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(true.flatten(), result.flatten())
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

ind=sum(threshold>0.5)
print("false positive rate="+str(fpr[ind]),"true positive rate="+str(tpr[ind]))