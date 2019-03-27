from flockAlgorithm import main, Bird

import tensorflow as tf
import keras
import tflearn
from tflearn.layers.core import input_data, fully_connected
import numpy as np

from random import randint
from statistics import mean
from collections import Counter
import pickle


class flockNN:
    def __init__ (self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'testData.txt'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
            [[-1, 0], 0],
            [[0, 1 ], 1],
            [[1, 0 ], 2],
            [[0, -1], 3]
            ]


    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 6, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def visualise_game(self, model):
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

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1,10,1)#input
        Y = np.array([i[1] for i in training_data]).reshape(-1,1)#output
        model.fit(X, Y, n_epoch = 100, shuffle=True, run_id = self.filename)##input data fed to train
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for i in range(self.test_games):
            steps = 0
            game_memory = []
            game = Bird()
            i, 

    def train(self):
        training_data = self.filename
        nn_model = self.model()
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    flockNN().train()
    #flockNN().visualise()