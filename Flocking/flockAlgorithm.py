import random
from math import sqrt, floor
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
        self.acceleration = [0.0 , 0.0]
        self.velocity = [self.forceX, self.forceY]
        self.maxSpeed = 1.0 ## max speed
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

    def ApplyForce(self, force=[]):
        self.acceleration+=force

    def calcSeperation(self):
        sepDist = 20.0
        steer = [0.0,0.0]
        distance = 0.0
        count = 0

        for i in Bird.flock:
            location = [self.posX, self.posY]

        for other in Bird.flock:
            if self != other:
                distance = sqrt(((self.posX-other.posX)**2)+((self.posY - other.posY)**2))##distance between
            if distance > 0.0 and distance < sepDist:
                diff = [self.location[0] - other.location[0], self.location[1] - other.location[1]]##subTwoLists
                mag = sqrt(diff[0]**2 + diff[1]**2)##magnitude
                if mag > 0:
                    diff[0]/=mag##normalize
                    diff[1]/=mag
                else:
                    diff[0] = diff[0]
                    diff[1] = diff[1]
                diff[0]/=distance##divScalar
                diff[1]/=distance
                steer = [self.posX+diff[0], self.posY+diff[1]]##addVector
                count+=1##increment count
        if count > 0:
            steer[0]/= count
            steer[1]/= count
        steerMag = sqrt(steer[0]**2 + steer[1]**2)##magnitude
        if steerMag > 0:
            if steerMag > 0:
                steer[0]/=steerMag##divScalar
                steer[1]/=steerMag
            else:
                steer[0] = diff[0]
                steer[1] = diff[1]
            steer = [steer[0]*self.maxSpeed, steer[1]*self.maxSpeed]##mulScalar
            steer = [steer[0]-self.forceX, steer[1] - self.forceY]
            if steerMag > self.maxForce:##limit
               self.posX = floor(self.posX / steerMag)
               self.posY = floor(self.posY / steerMag)
        return steer

    def calcAlignment(self):
        neighborDist = 50.0
        m_sum = [0.0,0.0]
        distance = 0.0
        count = 0
        for other in Bird.flock:
            if self != other:
                distance = sqrt(((self.posX-other.posX)**2)+((self.posY - other.posY)**2))
            if distance > 0 and distance < neighborDist:
                m_sum[0]+=self.velocity[0]
                m_sum[1]+=self.velocity[1]
                count+=1
            if count > 0:
                m_sum[0]/=count
                m_sum[1]/= count
                mag = sqrt(m_sum[0]**2 + m_sum[1]**2)##magnitude
                if mag > 0:
                    m_sum[0]/=mag##normalize
                    m_sum[1]/=mag
                else:
                    m_sum[0] = self.posX
                    m_sum[1] = self.posY
                m_sum = [m_sum[0]*self.maxForce, m_sum[1]*self.maxForce]
                steer[0,0]
                steer = [m_sum[0]- self.velocity[0], m_sum[1] - self.velocity[1]]
                steerMag = sqrt(steer[0]**2 + steer[1]**2)##magnitude
                if steerMag > self.maxForce:##limit
                    steer = [steer[0]/self.maxForce, steer[1]/self.maxForce]
                return steer
            else:
                temp = [0,0]
                return temp

    def calcCohesion(self):
        neighborDist = 50.0
        m_sum = [0.0,0.0]
        distance = 0.0
        count =0

        for other in Bird.flock:
            if self != other:
                distance = sqrt(((self.posX-other.posX)**2)+((self.posY - other.posY)**2))
            if distance > 0 and distance < neighborDist:
                m_sum = [self.posX + other.posX, self.posY+other.posY]
                count+=1
        if count > 0:
            m_sum = [m_sum[0]/count , m_sum[1]/count]
            return Bird.seek(self, m_sum)
        else:
            temp = [0.0,0.0]
            return temp

    def seek(self, m_sum, *args):
        desired = [0.0,0.0]
        desired = self.location
        desired = [desired[0] - m_sum[0], desired[1] - m_sum[1]]
        desiredMag = sqrt(desired[0]**2 + desired[1]**2)
        if desiredMag > 0:
            desired[0] /= desiredMag
            desired[1] /= desiredMag
        else:
            desired[0] = self.location[0]
            desired[1] = self.location[1]
        desired = [desired[0]*self.maxForce, desired[1]*self.maxForce]
        self.acceleration = desired
        accelMag = sqrt(self.acceleration[0]**2 + self.acceleration[1]**2)
        if accelMag > self.maxSpeed:
            self.acceleration = [self.acceleration[0]/self.maxForce, self.acceleration[1]/self.maxForce]
        
        return self.acceleration

    def update(self):
        self.acceleration = [self.acceleration[0]*.4, self.acceleration[1]*.4]
        self.velocity = [self.velocity[0]+self.acceleration[0], self.velocity[1]+self.acceleration[1]]
        velMag = sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if velMag > self.maxForce:
            self.velocity = [self.velocity[0]/self.maxForce, self.velocity[1]/self.maxForce]
        self.location = [self.location[0] + self.velocity[0], self.location[1] + self.velocity[1]]
        self.acceleration=[self.acceleration[0] * 0, self.acceleration[1]*0]
        Bird.render(self)

    def Flocking(self):
        sep = Bird.calcSeperation(self)
        ali = Bird.calcAlignment(self)
        coh = Bird.calcCohesion(self)

        sep = [sep[0]*1.5, sep[1]*1.5]
        ali = [ali[0]*1 ,ali[1]*1 ]
        coh = [coh[0]*1,coh[1]*1]

        self.ApplyForce(sep)
        self.ApplyForce(ali)
        self.ApplyForce(coh)

    def render(self):
        gameDisplay.blit(self.img, (self.posX,self.posY))

    def generate_observations(self):
        return self.alignment, self.cohesion, self.seperation

    def setHeading(self):
        pass

##game loop 
def main():
    flocking = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()##close window
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for i in range(len(Bird.flock)):
                        Bird.Flocking(Bird.flock[i])
                        if i >= 9:
                            i = 0
                    flocking = True

        if len(Bird.flock) < flockSize:
                Bird()
        else:
             for i in range(len(Bird.flock)):
                if flocking == False:
                    Bird.moveBird(Bird.flock[i])
                Bird.update(Bird.flock[i])
        gameDisplay.fill(blue)
        
        pygame.display.update()
        clock.tick(60)
        
main()