# coding: utf-8

# In[3]:


import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv  
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy
mnist = tf.keras.datasets.mnist #download minist


# In[4]:



(x_train, y_train),(x_test,y_test)=mnist.load_data()# split data 

x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)


# In[12]:


#cnn
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=3)
#As the number of dense increases the accuracy increases


# In[11]:



loss , accuracy  =model.evaluate(x_test,y_test)
print(accuracy)
print(loss)


# In[ ]:




