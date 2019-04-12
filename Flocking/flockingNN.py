import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from tflearn import optimizers
from tflearn.data_utils import VocabularyProcessor
import numpy as np
import pygame
from random import randint
from statistics import mean
from collections import Counter

from flockAlgorithm import Boid

##Create lists for inputs and outputs of training data
Input = []
Output = []

##Converts string to float and appends to Input list
with open('Inputs.txt') as I: ##create inputs from file
    line = I.readlines()
    for i in range(len(line)):
        line[i] = line[i].rstrip()#strip whitespace
        line[i] = line[i].replace("[", "")#remove square brakets
        line[i] = line[i].replace("]", "")
        Input.append(np.fromstring(line[i], dtype=float, sep = ','))#append to list
    I.close()


##Converts string to float and appends to Output list
with open('Outputs.txt') as O:
    Outputline = O.readlines()
    for i in range(len(line)):
        Outputline[i] = Outputline[i].rstrip()
        Output.append(np.fromstring(Outputline[i], dtype=float, sep = ','))
    print(Output[1])
    O.close()


class flockNN:
    def __init__ (self, lr = 0.1, filename = "./NeuralNet2.tflearn"): #constructor method for NN
        self.lr = lr
        self.train_Inputs = Input
        self.test_Inputs = Input
        self.train_Outputs = Output
        self.train_Outputs = Output
        self.filename = filename

    ##Neural net architecture
    def model(self):
        network = input_data(shape=[None, 5, 1], name='input') ## NN takes 5 inputs
        network = fully_connected(network, 25, activation='relu')#rectangular linear function for hidden layer
        network = fully_connected(network, 1, activation='linear')## linear output for continous numbers, Single output
        sgd = tflearn.optimizers.SGD(learning_rate=self.lr, lr_decay=0.096, decay_step=1000, staircase=False, use_locking=False, name='SGD') ## optimizer function with learning decay
        network = regression(network, optimizer=sgd, learning_rate=self.lr, loss='mean_square', name='target')#regression layer
        model = tflearn.DNN(network, tensorboard_dir='log')
        #model.load('./NeuralNet2.tflearn')
        return model

    ##Train NN
    def train_model(self, training_Inputs, training_Outputs, model):
        X = np.array([i for i in training_Inputs]).reshape(-1,5,1)#input
        Y = np.array([i for i in training_Outputs]).reshape(-1,1)#output
        model.fit(X, Y, n_epoch = 3, shuffle=True, run_id = './NeuralNet2.tflearn')##input data fed to train
        model.save('./NeuralNet2.tflearn')##Save NN checkpoint after training
        return model

    ##Test NN 
    def test_model(self,training_Inputs, training_Outputs, NN_Model):
        inputs = np.array([i for i in training_Inputs]).reshape(-1,5,1)##reshape inputs to fit NN
        outputs = np.array([i for i in training_Outputs]).reshape(-1,1)
        test_acc = NN_Model.evaluate(inputs, outputs)#Evaluate to find machines accuracy
        print('Test accuarcy: ', test_acc)
        prediction = NN_Model.predict(np.array([i for i in training_Inputs]).reshape(-1,5,1))#create predictions of what the machines think the output should be
        testPredict = NN_Model.predict(np.array([training_Inputs[8000]]).reshape(-1,5,1))
        print('testPredict: %s' % testPredict)
        print("Prediciton: %s" %prediction[10])#Write machines prediction
        print("Expected: "+ str(training_Outputs[10]))#Write machines expected output
        return prediction

    ##Calculate Prediction based on information fed into NN from Boid
    def test_Boid(self,Boid_testing=[]):
        Boid_Model = self.model()
        BoidInput = []
        Boid_testing = Boid_testing.replace("[", "")
        Boid_testing = Boid_testing.replace("]", "")
        BoidInput = (np.fromstring(Boid_testing, dtype=float, sep=','))
        X = np.array([i for i in BoidInput]).reshape(-1,5,1)
        print(X)
        BoidPrediction = Boid_Model.predict(X).reshape(-1,5,1)
        


        return BoidPrediction


    def train(self):
        training_Inputs = self.train_Inputs
        training_Outputs = self.train_Outputs
        nn_model = self.model()
        nn_model = self.train_model(training_Inputs, training_Outputs, nn_model)
        self.test_model(training_Inputs, training_Outputs, nn_model)

    ##Visualise the NN 
    def visualise(self):
        pygame.init()
        display_Width = 600 ## Display window dimensions
        display_Height = 600

        black=(0,0,0)
        White=(255,255,255) ##Colour presets

        gameDisplay = pygame.display.set_mode((display_Width,display_Height)) ##Create window
        pygame.display.set_caption("Flocking Window") ##Window Title
        gameDisplay.fill(black)
        clock = pygame.time.Clock()

        flockSize=8

        while True:
            gameDisplay.fill(White)
            if len(Boid.flock) < flockSize:
                    Boid()
            else:
                 for i in range(len(Boid.flock)):
                        information = Boid.NeuralNetFlocking(Boid.flock[i], Boid.flock)
                        Boid[i].rotation = flockNN.test_Boid(information)
                        Boid[i].heading[0] = cos(Boid[i].rotation * (3.14/180))
                        Boid[i].heading[1] = sin(Boid[i].rotation * (3.14/180))
                        Boid.update(Boid.flock[i])

                        if i >= len(Boid.flock):
                            i =0
            pygame.display.update()

    def test(self):
        nn_model = self.model()
        training_Inputs = self.train_Inputs
        training_Outputs = self.train_Outputs
        birdRot = self.test_model(training_Inputs, training_Outputs, nn_model)
        return birdRot


flockNN().train()
    #flockNN().visualise()