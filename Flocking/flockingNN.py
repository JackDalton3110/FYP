from flockAlgorithm import Bird
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from statistics import mean
import matplotlib.pyplot as plt
from collections import Counter

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