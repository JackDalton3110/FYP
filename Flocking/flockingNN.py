#from flockAlgorithm import main, Bird

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from tflearn import optimizers
from tflearn.data_utils import VocabularyProcessor
import numpy as np

from random import randint
from statistics import mean
from collections import Counter
Input = []
Output = []
print("tf version = "+tf.__version__)

with open('Inputs.txt') as I:
    line = I.readlines()
    for i in range(len(line)):
        line[i] = line[i].rstrip()
        line[i] = line[i].replace("[", "")
        line[i] = line[i].replace("]", "")
        Input.append(np.fromstring(line[i], dtype=float, sep = ','))
    I.close()

with open('Outputs.txt') as O:
    Outputline = O.readlines()
    for i in range(len(line)):
        Outputline[i] = Outputline[i].rstrip()
        Output.append(np.fromstring(Outputline[i], dtype=float, sep = ','))
    print(Output[1])
    O.close()


class flockNN:
    def __init__ (self, lr = 0.1, filename = "./NeuralNet2.tflearn"):
        self.lr = lr
        self.train_Inputs = Input
        self.test_Inputs = Input
        self.train_Outputs = Output
        self.train_Outputs = Output
        self.filename = filename

    def visualise_game(self, model):
        pass
        game = main()
        for i in range(self.goal_steps):
            predictions = []
            for action in range(-1, 2):
                predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1,5,1)))
            action = np.argmax(np.array(predictions))
            game_action = self.get_game_action(flock, action -1)
            done, _, snake, food  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(Bird, velocity)

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        sgd = tflearn.optimizers.SGD(learning_rate=self.lr, lr_decay=0.096, decay_step=1000, staircase=False, use_locking=False, name='SGD')
        network = regression(network, optimizer=sgd, learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        model.load('./NeuralNet2.tflearn')
        return model

    def train_model(self, training_Inputs, training_Outputs, model):
        X = np.array([i for i in training_Inputs]).reshape(-1,5,1)#input
        Y = np.array([i for i in training_Outputs]).reshape(-1,1)#output
        model.fit(X, Y, n_epoch = 5, shuffle=True, run_id = './NeuralNet2.tflearn')##input data fed to train
        model.save('./NeuralNet2.tflearn')
        return model

    def test_model(self,training_Inputs, training_Outputs, NN_Model):
        inputs = np.array([i for i in training_Inputs]).reshape(-1,5,1)
        outputs = np.array([i for i in training_Outputs]).reshape(-1,1)
        test_acc = NN_Model.evaluate(inputs, outputs)
        print('Test accuarcy: ', test_acc)
        prediction = NN_Model.predict(np.array([i for i in training_Inputs]).reshape(-1,5,1))
        print("Prediciton: %s" % str(prediction[10]))

    def test_Boid(self, NN_Model,Boid_testing):
        BoidInput = []
        Boid_testing = Boid_testing.replace("[", "")
        Boid_testing = Boid_testing.replace("]", "")
        BoidInput = (np.fromstring(Boid_testing, dtype=float, sep = ','))
        X = np.array([j for j in BoidInput]).reshape(1,5,1)
        BoidPrediction = NN_Model.predict(X)
        return BoidPrediction

    def train(self):
        training_Inputs = self.train_Inputs
        training_Outputs = self.train_Outputs
        nn_model = self.model()
        nn_model = self.train_model(training_Inputs, training_Outputs, nn_model)
        self.test_model(training_Inputs, training_Outputs, nn_model)

    def visualise(self):
        nn_model = self.model()
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        self.test_model(self.test_Inputs, self.test_Inputs, nn_model)

flockNN().train()
    #flockNN().visualise()