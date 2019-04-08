#from flockAlgorithm import main, Bird

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
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
    inputLine = Input[1]
    print(inputLine[1])
    I.close()
with open('Outputs.txt') as O:
    Outputline = O.readlines()
    for i in range(len(line)):
        Outputline[i] = Outputline[i].rstrip()
        Output.append(np.fromstring(Outputline[i], dtype=float, sep = ','))
    print(Output[1])
    O.close()

max_input_length = 6


class flockNN:
    def __init__ (self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 0.04, filename = "Inputs.txt"):
        self.lr = lr
        self.Inputs = Input
        self.Outputs = Output
        self.filename = filename


    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 200, activation='relu')
        #network = fully_connected(network, 21, activation='softmax')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

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

    def train_model(self, training_Inputs, training_Outputs, model):
        X = np.array([i for i in training_Inputs]).reshape(-1,5,1)#input
        print(X)
        Y = np.array([i for i in training_Outputs]).reshape(-1,1)#output
        print(Y)
        model.fit(X, Y, n_epoch = 1000, shuffle=True, run_id = self.filename)##input data fed to train
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []

    def train(self):
        training_Inputs = self.Inputs
        training_Outputs = self.Outputs
        nn_model = self.model()
        nn_model = self.train_model(training_Inputs, training_Outputs, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        self.test_model(nn_model)

#if __name__ == "__main__":
flockNN().train()
    #flockNN().visualise()