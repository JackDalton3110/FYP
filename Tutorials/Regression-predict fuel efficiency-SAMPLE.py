#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q seaborn')


# In[2]:


from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print (tf.__version__)


# In[3]:


dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path


# In[5]:


column_names = ['MPG', 'Cylinders', 'Displacement', 'HorsePower', 'Weight',
               'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                         na_values= "?", comment= '\t',
                         sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()


# # Clean the data
# The dataset contains a few unknown values

# In[7]:


dataset.isna().sum()


# In[8]:


dataset = dataset.dropna()


# In[9]:


origin = dataset.pop('Origin')


# In[10]:


dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()


# # Split the data into train and test
# Now split the data into a train and a test set.
# We will use the test set in the final evaluation of out model.

# In[11]:


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[13]:


sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")


# In[14]:


train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats


# # Split features from labels
# seperate the target value, or "label", from the features. This label is the value that you will train the model to predict.

# In[15]:


train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# # Normalize the data
# Look again at the train_stats block above and note how different the ranges of each feature are.
# 
# It is good practice to normalize features that use different scales and ranges. Although the model might converge without feature normalization, it makes training more difficult, and it makes the resulting model dependent on the choice of units used in the input.

# In[32]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# # Build the model
# Let's build our model. Here, we'll use a Sequential model with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, build_model, since we'll create a second model, later on.

# In[33]:


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    
    optimizer = tf.train.RMSPropOptimizer(0.001)
    
    model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae', 'mse'])
    return model


# In[34]:


model=build_model()


# # Inspect the model
# use the .summary method to print a simple description of the model

# In[35]:


model.summary()


# # Testing the model
# ### By taking a batch of 10 examples from the  training data and calling model.predict on them.

# In[36]:


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# # Training the model 
# The model is trained for 1000 epochs, and record the training and validation accuracy in the history object

# In[37]:


##Display training progress by printing a single dot for each complete epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print(' .', end='')
            
EPOCHS = 1000

history = model.fit(
            normed_train_data, train_labels,
            epochs=EPOCHS, validation_split = 0.2, verbose=0,
            callbacks=[PrintDot()])


# In[38]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[39]:


import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
            label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
            label = 'Val Error')
    plt.legend()
    plt.ylim([0,5])
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
            label = 'Val Error')
    plt.legend()
    plt.ylim([0,20])
    
plot_history(history)


# #### This graph shows little improvement, or even degradation in the validation error after a few hundred epochs. Let's update the model.fit method to automatically stop training when the validation score doesn't improve. We'll use a callback that tests a training condition for every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.

# In[40]:


model = build_model()

##The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                   validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# #### The graph shows that on the validation set, the average error usually around +/- 2 MPG. Is this good? We'll leave that decision up to you.
# 
# #### Let's see how did the model performs on the test set, which we did not use when training the model:

# In[42]:


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# # Make Predictions
# Finally, predict the MPG values using data in the testing set:

# In[43]:


test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_=plt.plot([-100,100], [-100,100])


# In[44]:


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error[MPG]")
_=plt.ylabel("Count")


# In[ ]:




