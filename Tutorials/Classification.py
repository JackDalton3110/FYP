#!/usr/bin/env python
# coding: utf-8

# In[30]:


from tensorflow import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
import numpy as np


# In[41]:


model = Sequential([
    Dense(16, input_shape=(2,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(2, activation='sigmoid')
])


# In[42]:


model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[43]:


#weight, height as input
train_samples = ((150, 67), (130,60), (200,65), (125, 53), (230, 72), (181,70))
train_samples = np.array(train_samples)


# In[44]:


#0: male
#1: female
#used to map which weight and height belongs to which gender
train_labels = [1,1,0,1,0,0]


# In[46]:


model.fit(x=train_samples, y=train_labels, batch_size=3, epochs=100, shuffle=True, verbose=2)


# In[ ]:




