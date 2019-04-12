#from flockAlgorithm import Boid
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from tflearn import optimizers
from tflearn.data_utils import VocabularyProcessor
import numpy as np
from math import sqrt, floor, atan2, pi, cos, sin
from random import randint
from statistics import mean
from collections import Counter
import pygame
import random


pygame.init()
display_Width = 600 ## Display window dimensions
display_Height = 600

black=(0,0,0)
White=(255,255,255) ##Colour presets
neuralFlockSize = 5
    
gameDisplay = pygame.display.set_mode((display_Width,display_Height))

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

class neuralFlock:
   

    flock = []

    def __init__(self):
        posX = float(random.randrange(display_Width))##randomise position for every Bird
        posY = float(random.randrange(display_Height))
        self.location = [posX, posY]
        self.rotation = float(round(random.randrange(360),2))
        self.heading = [0.0, 0.0]
        self.forceX = 0.0
        self.forceY = 0.0
        self.speed = [0.75, 0.75]
        self.acceleration = [0.0 , 0.0]
        self.velocity = [0.0, 0.0]
        self.img = pygame.image.load('./Flocking/images/flockImg.png')
        neuralFlock.flock.append(self)
    
    def moveBird(self):
        if self.forceX == 0.0 and self.forceY == 0.0:#randomise movement
            self.forceX = random.randrange(2.0)
            self.forceY = random.randrange(2.0)
            self.velocity = [self.forceX, self.forceY]
        self.location[0] += self.forceX
        self.location[1] += self.forceY
        self.location = [self.location[0], self.location[1]]
        gameDisplay.blit(self.img, (self.location[0],self.location[1]))
        self.rotation = float(random.randrange(360))#randomises rotation
        pygame.display.set_caption("Bird Moving Window")#changes title of screen
    
    def NeuralNetFlocking(self, flockList):
        closeEnough = 30
        extendedDist = 100
        nnRotation = []

        nnRotation.append(self.rotation)
        j=1

        for i in range(len(flockList)):
            distance = sqrt(((self.location[0]-flockList[i].location[0])**2)+((self.location[1] - flockList[i].location[1])**2))
            
            if(j == 5):
                break

            if(distance <= closeEnough and distance != 0):
                nnRotation.append(flockList[i].rotation)
                
                j+=1
            elif(distance <= extendedDist and distance != 0):
                nnRotation.append(flockList[i].rotation)
                j+=1
            else:
                nnRotation.append(self.rotation)
                j+=1
        
       
        InputArr = []
        X = str(nnRotation)
        X = X.rstrip()#strip whitespace
        X = X.replace("[", "")#remove square brakets
        X= X.replace("]", "")

        InputArr.append(np.fromstring(X, dtype=float, sep = ','))#append to list
        print(InputArr)
        #rotations = np.array([i for i in InputArr]).reshape(-1,5,1)

        return InputArr
    
    def update(self):
        self.velocity = [self.heading[0] * self.speed[0], self.heading[1] *self.speed[1]]##addVector
        self.location = [self.location[0] + self.velocity[0], self.location[1] + self.velocity[1]]
        neuralFlock.render(self)

    def borders(self):
        if(self.location[0] > display_Width):
            self.location[0] = 0
        elif(self.location[0] < 0):
            self.location[0] = display_Width

        if(self.location[1] < 0):
            self.location[1] = display_Height
        elif(self.location[1] > display_Height):
            self.location[1] = 0

    #Draws Boids
    def render(self):
        gameDisplay.blit(self.img, (self.location[0],self.location[1]))
    
   

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
        model.load('./NeuralNet2.tflearn')
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
        prediction = NN_Model.predict(np.array([i for i in training_Inputs]).reshape(-1,5,1))#create predictions of what the machines think the output should be
        print("Prediciton: %s" %prediction[10])#Write machines prediction
        print("Expected: "+ str(training_Outputs[10]))#Write machines expected output
        return prediction

    ##Calculate Prediction based on information fed into NN from Boid
    def test_Boid(self, Boid_testing, nn_model):
        X= np.array([i for i in Boid_testing]).reshape(-1,5,1)
        BoidPrediction = nn_model.predict(X)
        return BoidPrediction[0][0]


    def train(self):
        training_Inputs = self.train_Inputs
        training_Outputs = self.train_Outputs
        nn_model = self.model()
        nn_model = self.train_model(training_Inputs, training_Outputs, nn_model)
        self.test_model(training_Inputs, training_Outputs, nn_model)

    ##Visualise the NN 
    def visualise(self):
        
        pygame.display.set_caption("Neural Net Window") ##Window Title
        gameDisplay.fill(black)
        clock = pygame.time.Clock()
        NNmade = False
        while True:

            gameDisplay.fill(black)
            if len(neuralFlock.flock) < neuralFlockSize:
                    neuralFlock()
            elif len(neuralFlock.flock) == 5 and NNmade == False:
                    Boid_Model = flockNN().model()
                    NNmade = True
            elif NNmade == True:
                 for i in range(len(neuralFlock.flock)):
                        information = neuralFlock.NeuralNetFlocking(neuralFlock.flock[i], neuralFlock.flock)
                        print (sum(information)/len(information))
                        neuralFlock.flock[i].rotation = flockNN().testFlock(information, Boid_Model)
                        print(neuralFlock.flock[i].rotation)
                        neuralFlock.flock[i].heading[0] = cos(neuralFlock.flock[i].rotation * (3.14/180))
                        neuralFlock.flock[i].heading[1] = sin(neuralFlock.flock[i].rotation * (3.14/180))
                        neuralFlock.update(neuralFlock.flock[i])
                        neuralFlock.borders(neuralFlock.flock[i])
                        if i >= len(neuralFlock.flock):
                            i =0
            pygame.display.update()

    def test(self):
        nn_model = self.model()
        training_Inputs = self.train_Inputs
        training_Outputs = self.train_Outputs
        birdRot = self.test_model(training_Inputs, training_Outputs, nn_model)
        return birdRot

    def testFlock(self, information, Boid_Model):
        output = self.test_Boid(information, Boid_Model)
        return output


#flockNN().train()
flockNN().visualise()