from flockAlgorithm import Bird

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
from keras.optimizers import Adam
from tensorflow.keras import layers

import numpy as np

from random import randint

from collections import Counter
import pickle

tf.enable_eager_execution()

class flockNN:
    def __init__ (self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'flock_nn.tflearn'):
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

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = Bird()
            _, prev_score, Bird, flock = game.main()
            prev_observation = self.generate_observation(Bird, flock)
            prev_flock_distance = self.generate_flock_distance(Bird, flock)
            for i in range(self.goal_steps):
                action, game_action = self.generate_action(Bird)


    def model(self):
        network = input_data(shape[None,5,1], name='input')
        network = fully_connected(network,25,ativation='relu')
        network = fully_connected(network, 2,ativation='linear')
        netowrk = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def visualise_game(self, model):
        game = flockAlgorithm()
        prev_observation = self.generate_observation(distance, postion)
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
                prev_observation = self.generate_observation(bird, velocity)

    def train_model(self, training_data, model):
        pass

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_mdoel.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

    if __name__ == "main":
        flockNN().train()
        flockNN().visualise()


    