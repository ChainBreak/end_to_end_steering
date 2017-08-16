#!/usr/bin/env python

#from matplotlib import pyplot as plt
import cv2
import time
import numpy as np
import math
import keyboard


class Simulation:
    def __init__(self):
        self.view_h,self.view_w = (40,40)
        self.bg_h,self.bg_w = (100,100)
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
        self.x_input = 0
        self.y_input = 0
        self.dt = 0.1

    def getFrame(self):
        x = int(round(self.x))
        y = int(round(self.y))
        return self.bg_img[y:y+self.view_h, x:x+self.view_w]

    def keyboardInput(self,left,right,up,down):
        self.x_input = right - left
        self.y_input = up - down

    def update(self):
        #acceleration
        self.vx -= self.x_input * self.dt
        self.vy += self.y_input * self.dt

        #position update
        self.x += self.vx * self.dt * 10
        self.y += self.vy * self.dt * 10

        #position limits
        self.x = min(max(0,self.x), self.bg_w - self.view_w)
        self.y = min(max(0,self.y), self.bg_h - self.view_h)

        print("[%f, %f]"%(self.vx,self.vy))


sim = Simulation()

cv2.namedWindow("sim")
while True:

    sim.keyboardInput(  keyboard.is_pressed("left"),
                        keyboard.is_pressed("right"),
                        keyboard.is_pressed("up"),
                        keyboard.is_pressed("down"))
    sim.update()

    f = sim.getFrame()
    cv2.imshow("sim",f)
    # plt.clf()
    # plt.imshow(f)
    # plt.show(block=False)
    # plt.pause(0.001)
    time.sleep(1)

    if keyboard.is_pressed('esc'):
       break
plt.close()
