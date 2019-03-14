#!/usr/bin/env python
# coding: utf-8

# ## Train your first neural network

# In[1]:


##
import tensorflow as tf
from tensorflow import keras

##Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# ## Import MNIST dataset

# In[2]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# ## Class Names

# In[3]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[4]:


train_images.shape


# In[5]:


len(train_labels)


# In[6]:


train_labels


# In[7]:


test_images.shape


# In[8]:


len(test_labels)


# ## Preprocess the data
# the data must be preprocessed before the training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

# In[10]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)


# In[11]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[13]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# ## Setting up the layers
# Most deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, have parameters that are learned during training. 

# In[15]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# tf.keras.layers.Flatten is used to transform the format of the image from a 2d array to a 1d array. This layer has no parameter to learn; it only reformats the data.

# # Compile the model
# before the model is ready for training more settings must be added. This is done while compiling the model.
# 
# ### Loss function:
# measures the accuracy of the model during trianing. The goal is to minimize how "incorrect" the model is, by using this we can help the model learn in the right direction.
# 
# ### Optimizer:
# Updates the model based on the data and loss function
# 
# ### Metrics:
# Used to monitor the training and testing steps.

# In[16]:


model.compile(optimizer=tf.train.AdamOptimizer(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# # Training the Model
# Training the nueral network model requires the following steps:
# 
# ## 1:
# Feed the training data to the model - in this example, the train_images and train_labels arrays.
# 
# ## 2:
# The model learns to associated images and labels.
# 
# ## 3:
# We ask the model to make predictions about a test set - in this example, the test_images array. We verify that the predictions match the labels from the test_labels array. 
# 
# To start training, call the model.fit method - the model is "fit" to the training data:

# In[18]:


model.fit(train_images, train_labels, epochs=30)


# # Evaluate accuracy
# Next compare how the model performs on the test dataset:

# In[19]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)


# # Make predictions
# With the model trained, we cann use it to make predictions about some images

# In[20]:


predictions = model.predict(test_images)


# In[22]:


predictions[5]


# In[24]:


np.argmax(predictions[5])


# In[25]:


test_labels[5]


# In[31]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[41]:


i = 11
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)


# In[44]:


## Plot the first X test image, their predicted label, and the true label
## Color correct predictions in blue, incorrect predictions

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)


# # Using the trained model to make a prediciton of a single image

# In[45]:


##Grab an image from the test dataset
img = test_images[5]
print(img.shape)


# #### tf.keras models are optimized to make predicitons on a batch, or collection, of examples at once. So even though we're using a single image, we need to add it to a list:

# In[46]:


##Add the image to a batch where it's the only member.

img = (np.expand_dims(img, 0))
print(img.shape)


# #### Now time to predict the image

# In[47]:


predictions_single = model.predict(img)
print(predictions_single)


# In[48]:


plot_value_array(0, predictions_single, test_labels)
_=plt.xticks(range(10), class_names, rotation=45)


# #### model.predict returns a list of lists, one for each image in the batch of data. Grab the predicition for our (only) image in the batch: 

# In[53]:


np.argmax(predictions_single[0])


# In[ ]:





# In[ ]:




