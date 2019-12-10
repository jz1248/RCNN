#!/usr/bin/env python
# coding: utf-8


# In[3]:


import os,cv2, keras

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping
# In[4]:


path = "Images"


cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


# In[8]:


model_loaded = keras.models.load_model('ieeercnn_vgg16_1.h5')
model_loaded.summary()

hist = pickle.load(open('history.dat', 'rb'))

import matplotlib.pyplot as plt
# plt.plot(hist.history["acc"])
# plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss","Validation Loss"])
plt.savefig('chart loss.png')
plt.show()



# In[ ]:

X_test = pickle.load(open('X_test.dat', 'rb'))

im = X_test[1600]
plt.imshow(im)
img = np.expand_dims(im, axis=0)
out= model_loaded.predict(img)
if out[0][0] > out[0][1]:
    print("plane")
else:
    print("not plane")
plt.show()

z=0
for e,i in enumerate(os.listdir(path)):
    if i.startswith("4"):
        z += 1
        img = cv2.imread(os.path.join(path,i))
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.copy()
        for e,result in enumerate(ssresults):
            if e < 2000:
                x,y,w,h = result
                timage = imout[y:y+h,x:x+w]
                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out= model_loaded.predict(img)
                if out[0][0] > 0.65:
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(imout)
        plt.show()

