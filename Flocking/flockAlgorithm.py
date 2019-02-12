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

class Boid(bird):
    flock = []

    def __init__(self):
        bird.__init__(self)
        self.up
        self.setheading(random.randrange(360))
        self.setpos(random.randrange(-100,200),random.randrange(-100,200))
        self.down()
        self.newHead = None
        Boid.flock.append(self)

pygame.draw.line(gameDisplay,black,(100,200),(100,700),2)
pygame.draw.polygon(gameDisplay,green,((10,20),(10,50),(30,25)))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    pygame.display.update()
