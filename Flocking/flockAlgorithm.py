import random
from math import *
from vectorForm import vector as vc
import pygame
pygame.init()



m_Radius = 20

white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
flockSize = 10

display_Width = 800
display_Height = 600

gameDisplay = pygame.display.set_mode((display_Width,display_Height))
pygame.display.set_caption("flocking algorithm")
gameDisplay.fill(blue)
clock = pygame.time.Clock()

##Flock members
class Bird:

    flock = []

    def __init__(self, gui = False):
        self.posX = random.randrange(display_Width)
        self.posY = random.randrange(display_Height)
        self.location = [self.posX, self.posY]
        self.seperation = 0.0
        self.alignment = 0.0
        self.cohesion = 0.0
        self.forceX = random.randrange(-1.0, 2.0)
        self.forceY = random.randrange(-1.0, 2.0)
        self.acceleration = [0.0,0.0]
        self.velocity = [self.forceX, self.forceY]
        self.maxSpeed = 1.0 ##
        self.maxForce = 0.5 ## steering force
        self.newHead = None
        self.gui = gui
        self.img = pygame.image.load('C:/Users/gameuser/Desktop/FYP/Flocking/images/flockArrow.png')
        Bird.flock.append(self)

    def start(self):
        pass
        self.setHeading()
        return self.generate_observations()

    def dist(self):
        pass
        ##for other in range(len(Bird.flock)):


    def moveBird(self):
        pass
        for i in range (len(Bird.flock)):
            self.posX += self.forceX
            self.posY += self.forceY
            self.location = [self.posX, self.posY]
            gameDisplay.blit(self.img, (self.posX,self.posY))

            if(self.posX > display_Width):
                self.posX = 0
            elif(self.posX < 0):
                self.posX = display_Width

            if(self.posY<0):
                self.posY = display_Height
            elif(self.posY>display_Height):
                self.posY = 0


    def calcCohesion(self):
        pass

    

    def calcSeperation(self):
        pass
        sepDist = 20
        steer = [0,0]
        count = 0

        for i in range (len(Bird.flock)):
            location = [self.posX, self.posY]

        for other in Bird.flock:
            if self != other:
                distance = sqrt(((self.posX-other.posX)**2)+((self.posY - other.posY)**2))
            if distance > 0 and distance < sepDist:
                diff = [self.location[0] - other.location[0], self.location[1] - other.location[1]]
                mag = sqrt(diff[0]**2 + diff[1]**2)
                if mag > 0:
                    diff[0]/=mag
                    diff[1]/=mag
                else:
                    diff[0] = diff[0]
                    diff[1] = diff[1]
                diff[0]/=distance
                diff[1]/=distance
                steer = [diff[0], diff[1]]
                count+=1
        if count > 0:
            steer[0]/= count
            steer[1]/= count
        steerMag = sqrt(steer[0]**2 + steer[1]**2)
        if steerMag > 0:
            if steerMag > 0:
                steer[0]/=steerMag
                steer[1]/=steerMag
            else:
                steer[0] = diff[0]
                steer[1] = diff[1]
            steer = [steer[0]*self.maxSpeed, steer[1]*self.maxSpeed]
            steer = [steer[0]-self.forceX, steer[1] - self.forceY]
            if steerMag > 1:
               self.posX = floor(self.posX / steerMag)
               self.posY = floor(self.posY / steerMag)
            return steer

    def calcAlignment(self):
        pass
        neighborDist = 50
        m_sum = [0,0]
        count = 0
        for other in range(len(Bird.flock)):
            if self != other:
                distance = sqrt(((self.posX-other.posX)**2)+((self.posY - other.posY)**2))
            if distance > 0 and d<neighborDist:
                m_sum[0]+=self.velocity[0]
                m_sum[1]+=self.velocity[1]
                count+=1
            if count > 0:
                m_sum
                

    def Flocking(self):
        pass

    def generate_observations(self):
        return self.alignment, self.cohesion, self.seperation

    def setHeading(self):
        pass

    def update(self):
        pass

##game loop 
def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()##close window
                quit()
        if len(Bird.flock) < flockSize:
                Bird()

        gameDisplay.fill(blue)
        for i in range(len(Bird.flock)):
            Bird.moveBird(Bird.flock[i])
            if i >= 9:
                i = 0
        
        pygame.display.update()
        clock.tick(60)
        
main()