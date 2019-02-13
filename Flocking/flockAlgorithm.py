import pygame
import random
from math import cos, radians
pygame.init()

blue = (0,0,255)
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
green = (0,255,0)

display_Width = 800
display_Height = 600

gameDisplay = pygame.display.set_mode((display_Width,display_Height))
pygame.display.set_caption("flocking algorithm")
gameDisplay.fill(blue)
clock = pygame.time.Clock()

pixAr = pygame.PixelArray(gameDisplay)
pixAr[10][20] = green

pygame.draw.polygon(gameDisplay,green,((posX,posY),(posX,posY+30),(posX+20,posY+25)))

class Boid:
    flock = []

    def __init__(self):
        bird.__init__(self)
        self.up
        self.setheading(random.randrange(360))
        posX = random.randrange(-100,200)
        posY = random.randrange(-100,200)
        self.setpos(posX,posY)
        self.down()
        self.newHead = None
        Boid.flock.append(self)
        pygame.draw.polygon(gameDisplay,green,((posX,posY),(posX,posY+30),(posX+20,posY+25)))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    pygame.display.update()
