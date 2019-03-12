import random
from math import sqrt, floor
import pygame
pygame.init()

m_Radius = 10

white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
flockSize = 10

display_Width = 1000
display_Height = 900

posX = float(random.randrange(display_Width))
posY = float(random.randrange(display_Height))

gameDisplay = pygame.display.set_mode((display_Width,display_Height))
pygame.display.set_caption("flocking algorithm")
gameDisplay.fill(blue)
clock = pygame.time.Clock()

##Flock members
class Bird:

    flock = []

    def __init__(self, gui = False):
        self.location = [posX, posY]
        self.seperation = 0.0
        self.alignment = 0.0
        self.cohesion = 0.0
        self.forceX = random.randrange(-1.0, 2.0)
        self.forceY = random.randrange(-1.0, 2.0)
        self.acceleration = [0.0 , 0.0]
        self.velocity = [self.forceX, self.forceY]
        self.maxSpeed = 0.75 ## max speed
        self.maxForce = 0.5 ## steering force
        self.newHead = None
        self.leader = False
        self.gui = gui
        self.img = pygame.image.load('C:/Users/gameuser/Desktop/FYP/Flocking/images/flockArrow.png')
        Bird.flock.append(self)

    def start(self):
        return self.generate_observations()

    def moveBird(self):
        if self.forceX and self.forceY:
            self.forceX = random.randrange(-1.0, 2.0)
            self.forceY = random.randrange(-1.0, 2.0)
            self.velocity = [self.forceX, self.forceY]
        self.location[0] += self.forceX
        self.location[1] += self.forceY
        self.location = [self.location[0], self.location[1]]
        gameDisplay.blit(self.img, (self.location[0],self.location[1]))
        
        if(self.location[0] > display_Width):
            self.location[0] = 0
        elif(self.location[0] < 0):
            self.location[0] = display_Width

        if(self.location[1]<0):
            self.location[1] = display_Height
        elif(self.location[1]>display_Height):
            self.location[1] = 0

    def ApplyForce(self, force=[]):
        self.acceleration = [self.acceleration[0]+force[0], self.acceleration[1]+force[1]]

    def calcSeperation(self, flockList):
        sepDist = 20.0
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
                ##steer = [self.location[0]+diff[0], self.location+diff[1]]##addVector
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

    def calcAlignment(self, flockList):
        neighborDist = 50.0
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

    def calcCohesion(self, flockList):
        neighborDist = 50.0
        m_sum = [0.0,0.0]
        distance = 0.0
        count =0

        for i in range(len(flockList)):
            distance = sqrt(((self.location[0]-flockList[i].location[0])**2)+((self.location[1] - flockList[i].location[1])**2))
            if distance > 0 and distance < neighborDist:
                m_sum = [self.location[0] + flockList[i].location[0], self.location[1]+flockList[i].location[1]]
                count+=1
        if count < 0:
            m_sum = [m_sum[0] / count , m_sum[1] / count]
            return Bird.seek(self, m_sum)
        else:
            temp = [0.0,0.0]
            return temp

    def seek(self, m_sum, *args):
        desired = [0.0,0.0]
        if self.leader == True:
            desired = self.location
        desired = [desired[0] - m_sum[0], desired[1] - m_sum[1]]
        desiredMag = sqrt(desired[0]**2 + desired[1]**2)
        if desiredMag > 0:
            desired[0] /= desiredMag
            desired[1] /= desiredMag
        else:
            desired[0] = self.desired[0]
            desired[1] = self.desired[1]
        desired = [desired[0]*self.maxSpeed, desired[1] * self.maxSpeed]##mulscalar
        self.acceleration = desired
        accelMag = sqrt(self.acceleration[0]**2 + self.acceleration[1]**2)
        if accelMag > self.maxSpeed:
            self.acceleration = [self.acceleration[0]/self.maxForce, self.acceleration[1]/self.maxForce]
        return self.acceleration

    def update(self):
        self.acceleration = [self.acceleration[0]*.4, self.acceleration[1]*.4]##mulScalar
        self.velocity = [self.velocity[0]+self.acceleration[0], self.velocity[1]+self.acceleration[1]]##addVector
        velMag = sqrt(self.velocity[0]**2 + self.velocity[1]**2)##magnitude
        if velMag > self.maxSpeed:##limit
            self.velocity = [self.velocity[0] / velMag, self.velocity[1] / velMag]
        self.location = [self.location[0] + self.velocity[0], self.location[1] + self.velocity[1]]
        self.acceleration=[self.acceleration[0] * 0, self.acceleration[1]*0]
        Bird.render(self)
        Bird.borders(self)
        Bird.flock[0].leader = True
        Bird.flock[0].img = pygame.image.load('C:/Users/gameuser/Desktop/FYP/Flocking/images/leaderArrow.png')
       # if self.forceX == 0.0 and self.forceY == 0.0:
          #  self.forceX = random.randrange(-1.0, 2.0)
          #  self.forceY = random.randrange(-1.0, 2.0)
          #  self.velocity = [self.forceX, self.forceY]


    def borders(self):
        if(self.location[0] > display_Width):
            self.location[0] = 0
        elif(self.location[0] < 0):
            self.location[0] = display_Width

        if(self.location[1] < 0):
            self.location[1] = display_Height
        elif(self.location[1] > display_Height):
            self.location[1] = 0

    def Flocking(self, flockList):
        closeEnough = 40
        sep = Bird.calcSeperation(self, Bird.flock)
        align = Bird.calcAlignment(self, Bird.flock)
        coh = Bird.calcCohesion(self, Bird.flock)

        sep = [sep[0]*1.5, sep[1]*1.5]
        align = [align[0]*1 ,align[1]*1 ]
        coh = [coh[0]*1,coh[1]*1]

        self.ApplyForce(sep)
        self.ApplyForce(align)
        self.ApplyForce(coh)

        for i in range(len(flockList)):
            distance = sqrt(((self.location[0]-flockList[i].location[0])**2)+((self.location[1] - flockList[i].location[1])**2))
        if distance <= closeEnough and self.leader == False:
            self.velocity = Bird.flock[0].velocity
        elif distance >= closeEnough and self.leader == False:
            m_sum = [self.location[0] + flockList[i].location[0], self.location[1] + flockList[i].location[1]]
            Bird.seek(self, m_sum)

    def render(self):
        gameDisplay.blit(self.img, (self.location[0],self.location[1]))

    def generate_observations(self):
        return self.alignment, self.cohesion, self.seperation

    def setHeading(self):
        pass

##game loop 
def main():
    flocking = True
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

        gameDisplay.fill(blue)
        if len(Bird.flock) < flockSize:
                Bird()
        else:
             for i in range(len(Bird.flock)):
                if flocking == False:
                    Bird.moveBird(Bird.flock[i])
                elif flocking == True:
                    for i in range(len(Bird.flock)):
                        Bird.Flocking(Bird.flock[i], Bird.flock)
                        Bird.update(Bird.flock[i])
                        if i >= 9:
                            i = 0
            
        
        pygame.display.update()
        clock.tick(60)
        
main()