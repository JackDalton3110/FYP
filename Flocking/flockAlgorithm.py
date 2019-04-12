import random
from math import sqrt, floor, atan2, pi, cos, sin
import pickle
import pygame
import numpy as np
pygame.init()

#neuralNet = flockNN()

white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
flockSize = 15

nearestRotation = [0.0, 0.0 ,0.0 ,0.0, 0.0]

display_Width = 1200
display_Height = 600

gameDisplay = pygame.display.set_mode((display_Width,display_Height))
pygame.display.set_caption("Flocking Window")
gameDisplay.fill(black)
clock = pygame.time.Clock()

##Flock members
class Boid:

    flock = []

    ##Constructor Function
    def __init__(self, gui = False):
        posX = float(random.randrange(display_Width))##randomise position for every Bird
        posY = float(random.randrange(display_Height))
        self.location = [posX, posY]
        self.seperation = 0.0
        self.alignment = 0.0
        self.cohesion = 0.0 
        self.rotation = float(round(random.randrange(360),2))
        self.heading = [0.0, 0.0]
        self.speed = [0.75, 0.75]
        self.acceleration = [0.0 , 0.0]
        self.forceX = 0.0
        self.forceY = 0.0
        self.velocity = [0.0, 0.0]
        self.maxSpeed = 2.0 ## max speed
        self.maxForce = 0.5 ## steering force
        self.img = pygame.image.load('./Flocking/images/flockImg.png')
        self.outputRotation = 0.0
        Boid.flock.append(self)

    ##Moves Boids randomly and randomises their rotation
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

    ##Takes force argument and applies that to Boids
    def ApplyForce(self, force=[]):
        self.acceleration = [self.acceleration[0]+force[0], self.acceleration[1]+force[1]]


     ##Calculate Seperation and return to Boid
    def calcSeperation(self, flockList):
        sepDist = 100.0
        steer = [0.0,0.0]
        distance = 0.0
        count = 0

        for i in range(len(flockList)):
            distance = sqrt(((self.location[0]-flockList[i].location[0])**2)+((self.location[1] - flockList[i].location[1])**2))##distance between
            if distance > 0.0 and distance < sepDist:
                diff = [float(self.location[0] - flockList[i].location[0]), float(self.location[1] - flockList[i].location[1])]##subTwoLists
                mag = sqrt(diff[0]**2 + diff[1]**2)##magnitude
                if mag > 0:
                    diff[0]/=mag##normalize
                    diff[1]/=mag
                else:
                    diff[0] = diff[0]
                    diff[1] = diff[1]
                diff[0]/=distance##divScalar
                diff[1]/=distance
                self.location[0]+=diff[0]
                self.location[1]+=diff[1]
                steer = [self.location[0], self.location[1]]
                count+=1##increment count

        if count > 0:
            steer[0]/= count
            steer[1]/= count
        steerMag = sqrt(steer[0]**2 + steer[1]**2)##magnitude
        if steerMag > 0:
            if steerMag > 0:
                steer = [steer[0]/steerMag, steer[1]/steerMag]##normalize
            else:
                steer = [steer[0],steer[1]]
                
            steer = [steer[0]*self.maxSpeed, steer[1]*self.maxSpeed]##mulScalar
            steer = [steer[0]-self.velocity[0], steer[1] - self.velocity[1]]
            if steerMag > self.maxForce:##limit
               steer = [steer[0]/steerMag, steer[1]/steerMag]
        return steer
    
    ##Calculate Alignment and return to Boid
    def calcAlignment(self, flockList):
        neighborDist = 10.0
        m_sum = [0.0,0.0]
        distance = 0.0
        count = 0
        steer = [0,0]

        for i in range(len(flockList)):
            distance = sqrt(((self.location[0]-flockList[i].location[1])**2)+((self.location[1] - flockList[i].location[1])**2))
            if distance > 0 and distance < neighborDist:
                m_sum = [m_sum[0]+self.velocity[0], m_sum[1]+self.velocity[1]]
                count+=1

        if count > 0:
            m_sum[0]/=count
            m_sum[1]/= count
            mag = sqrt(m_sum[0]**2 + m_sum[1]**2)##magnitude
            if mag > 0:
                m_sum[0]/=mag##normalize
                m_sum[1]/=mag
            else:
                m_sum[0] = self.location[0]
                m_sum[1] = self.location[1]
            m_sum = [m_sum[0]*self.maxForce, m_sum[1]*self.maxForce]
            steer = [m_sum[0] - self.velocity[0], m_sum[1] - self.velocity[1]]
            steerMag = sqrt(steer[0]**2 + steer[1]**2)##magnitude
            if steerMag > self.maxForce:##limit
                steer = [steer[0]/steerMag, steer[1]/steerMag]
            return steer
        else:
            temp = [0.0,0.0]
            return temp

    ##Calculate cohesion and return value
    def calcCohesion(self, flockList):
        neighborDist = 20.0
        m_sum = [0.0,0.0]
        distance = 0.0
        count =0

        for i in range(len(flockList)):
            distance = sqrt(((self.location[0]-flockList[i].location[0])**2)+((self.location[1] - flockList[i].location[1])**2))
            if distance > 0 and distance < neighborDist:
                m_sum = [self.location[0] + flockList[i].location[0], self.location[1]+flockList[i].location[1]]
                count+=1

        if count > 0:
            m_sum = [m_sum[0] / count , m_sum[1] / count]
            return Boid.seek(self, m_sum)
        else:
            temp = [0.0,0.0]
            return temp

    #seek meethod to move closer to other Boids
    def seek(self, m_sum, *args):
        desired = self.location
        desired = [desired[0] - m_sum[0], desired[1] - m_sum[1]]
        desiredMag = sqrt(desired[0]**2 + desired[1]**2)
        if desiredMag > 0:
            desired[0] /= desiredMag
            desired[1] /= desiredMag
        else:
            desired[0] = desired[0]
            desired[1] = desired[1]
        desired = [desired[0]*self.maxSpeed, desired[1] * self.maxSpeed]##mulscalar
        self.acceleration = desired
        accelMag = sqrt(self.acceleration[0]**2 + self.acceleration[1]**2)
        if accelMag > self.maxSpeed:
            self.acceleration = [self.acceleration[0]/self.maxForce, self.acceleration[1]/self.maxForce]
        return self.acceleration
    

    ## Calculates average rotation based on nearest 5 rotations
    ## Uses new rotation to calculate heading and applies heading to the Boid
    def calcHeading(self, flockList):
        closeEnough = 30
        extendedDist = 100
        nearestRotation[0] = self.rotation
        j=1
        
        #self.heading[0] = cos(self.rotation * (3.14/180))
        #self.heading[1] = sin(self.rotation * (3.14/180))

        for i in range(len(flockList)):
            distance = sqrt(((self.location[0]-flockList[i].location[0])**2)+((self.location[1] - flockList[i].location[1])**2))
            
            if(j == 5):
                break

            if(distance <= closeEnough and distance != 0):#if within distance add rotation to list
                nearestRotation[j]= flockList[i].rotation
                j+=1
            elif(distance <= extendedDist and distance != 0):
                nearestRotation[j] = flockList[i].rotation
                j+=1
            else:
                nearestRotation[j] = self.rotation#else add own rotation
                j+=1
        
        self.rotation = round(sum(nearestRotation)/len(nearestRotation),2)#calculate x and y heading
        self.heading[0] = cos(self.rotation * (3.14/180))
        self.heading[1] = sin(self.rotation * (3.14/180))

        #self.outputRotation = self.rotation # save rotation 

    ##Predict new rotation after feeding the closest 5 rotation to the ANN
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
        rotations = np.array([i for i in InputArr]).reshape(-1,5,1)

        

        return rotations


    ##update method
    def update(self):
        self.velocity = [self.heading[0] * self.speed[0], self.heading[1] *self.speed[1]]##addVector
        self.location = [self.location[0] + self.velocity[0], self.location[1] + self.velocity[1]]
        Boid.render(self)

    ##Makes the game enviroment a wrap around
    def borders(self):
        if(self.location[0] > display_Width):
            self.location[0] = 0
        elif(self.location[0] < 0):
            self.location[0] = display_Width

        if(self.location[1] < 0):
            self.location[1] = display_Height
        elif(self.location[1] > display_Height):
            self.location[1] = 0

    ##Calculates the seperation, cohesion and alignment then applies force to Boid
    def Flocking(self, flockList):

        sep = Boid.calcSeperation(self, flockList)
        align = Boid.calcAlignment(self, flockList)
        coh = Boid.calcCohesion(self, flockList)

        sep = [sep[0]*1.5, sep[1]*1.5]
        align = [align[0]*1 ,align[1]*1 ]
        coh = [coh[0]*1,coh[1]*1]

        self.ApplyForce(sep)
        self.ApplyForce(align)
        self.ApplyForce(coh)
        pygame.display.set_caption("Flocking Window")

    #Draws Boids
    def render(self):
        gameDisplay.blit(self.img, (self.location[0],self.location[1]))


##Used to record training data
##Records closest 5 rotations to one file (Inputs)
##Records rotation after calculating the average (Output)
def writeToFile(output, input=[]):
    filename = 'Inputs.txt'
    file = open(filename, "a")
    inputs = str( input)+'\n'
    file.write(inputs)

    filename = 'Outputs.txt'
    file = open(filename, 'a')
    Outputs = str(output)+'\n'
    file.write(Outputs)
    file.close()

##game loop 
def main():
    flocking = True
    neuralFlock= False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()##close window
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and flocking == True:
                    flocking = False
                else:
                    flocking = True
                if event.key == pygame.K_a and flocking == True and neuralFlock == False:
                    neuralFlock = True
                    flocking = False
                else:
                    neuralFlock = False
                    flocking = False

        gameDisplay.fill(black)
        if len(Boid.flock) < flockSize:
                Boid() ##Fill flock with Boid objects
        else:
             for i in range(len(Boid.flock)):
                if flocking == False and neuralFlock == False:
                    Boid.moveBird(Boid.flock[i])
                    Boid.borders(Boid.flock[i])
                elif flocking == True and neuralFlock == False:
                    for i in range(len(Boid.flock)):
                        Boid.Flocking(Boid.flock[i], Boid.flock)
                        Boid.borders(Boid.flock[i])
                        Boid.calcHeading(Boid.flock[i], Boid.flock)
                        Boid.update(Boid.flock[i])
                        #writeToFile(Bird.flock[i].outputRotation,nearestRotation)
                        if i >= len(Boid.flock):
                            i = 0
                elif neuralFlock == True:
                    for i in range(len(Boid.flock)):
                        Boid.NeuralNetFlocking(Boid.flock[i], Boid.flock)
                        Boid.update(Boid.flock[i])
                        if i >= len(Boid.flock):
                            i =0
        
        pygame.display.update()
        
main()