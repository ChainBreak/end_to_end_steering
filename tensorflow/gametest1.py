#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
import math
import keyboard



running = True

def checkKeys():
    key = ord(getch())
    if key == 27:
        keyPressed("esc")
    elif key != 255:
        print(key)


def keyPressed(key):
    if key == "esc":
        running = False



class Simulation:
    def __init__(self):
        self.view_h,self.view_w = (50,50)
        self.bg_h,self.bg_w = (400,400)
        self.bg_img = np.zeros([self.bg_h, self.bg_w])
        max_d = math.sqrt(self.bg_h**2 + self.bg_w**2) / 2
        cx = self.bg_w/2
        cy = self.bg_h/2
        for y in range(self.bg_h):
            for x in range(self.bg_w):
                self.bg_img[y,x] = 1.0 / ( 1.0 + (math.sqrt((y - cy)**2 + (x - cx)**2 ) )/max_d)

        self.x = (self.bg_w - self.view_w ) /2
        self.y = (self.bg_h - self.view_h ) /2
        self.vx = 0.0
        self.vy = 0.0

    def getFrame(self):
        x = int(round(self.x))
        y = int(round(self.y))
        return self.bg_img[y:y+self.view_h, x:x+self.view_w]

    def update(self):
        pass

sim = Simulation()

plt.imshow(sim.getFrame())
plt.show()

while running:
   if keyboard.is_pressed('up'):
       running = False
