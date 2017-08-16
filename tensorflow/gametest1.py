#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf
import math
import pygame


class Simulation:
    def __init__(self):
        self.view_h,self.view_w = (40,40)
        self.bg_h,self.bg_w = (80,80)
        self.bg_img = np.zeros([self.bg_h, self.bg_w])
        self.max_d = math.sqrt(self.bg_h**2 + self.bg_w**2) / 2
        cx = self.bg_w/2
        cy = self.bg_h/2
        for y in range(self.bg_h):
            for x in range(self.bg_w):
                d = math.sqrt((y - cy)**2 + (x - cx)**2 )
                self.bg_img[y,x] = 1.0 / ( 1.0 + d**1.8/self.max_d)

        self.x = (self.bg_w - self.view_w ) /3
        self.y = (self.bg_h - self.view_h ) /3
        self.x_input = 0
        self.y_input = 0
        self.speed = 50 #pixels per second
        self.dt = 0.1

    def getFrame(self):
        x = int(round(self.x))
        y = int(round(self.y))
        return self.bg_img[y:y+self.view_h, x:x+self.view_w].copy()

    def input(self,x_input,y_input):
        self.x_input = x_input
        self.y_input = y_input

    def update(self,deltaTime):

        #add unstable distance to the position
        cx = (self.bg_w - self.view_w)/2
        cy = (self.bg_h - self.view_h)/2
        max_d = math.sqrt(cx**2 + cy**2)
        cdx = self.x - cx #distance from center in x
        cdy = self.y - cy #distance from center in y
        cd = math.sqrt(cdx**2 + cdy**2 )+0.0001
        nx = cdx/cd
        ny = cdy/cd
        disturbance_x = nx * cd / max_d * self.speed / 3
        disturbance_y = ny * cd / max_d * self.speed / 3
        #print(max_d)
        #print("accuracy %i%% "%((1-cd/max_d)*100))

        #add the disturbance to the position
        self.x += disturbance_x * deltaTime
        self.y += disturbance_y * deltaTime

        #position update
        self.x -= self.x_input * deltaTime * self.speed
        self.y += self.y_input * deltaTime * self.speed

        #position limits
        self.x = min(max(0,self.x), self.bg_w - self.view_w)
        self.y = min(max(0,self.y), self.bg_h - self.view_h)


class InputSmoother:
    """up is positive, right is positive"""
    def __init__(self,timeToFullSpeed):
        self.timeToFullSpeed = timeToFullSpeed
        self.x_smooth = 0.0
        self.y_smooth = 0.0
        self.x_raw = 0.0
        self.y_raw = 0.0


    def keyboardInput(self,up,down,left,right):
        self.x_raw = right - left
        self.y_raw = up - down

    def update(self,deltaTime):
        x_step = 1.0/self.timeToFullSpeed*deltaTime
        y_step = 1.0/self.timeToFullSpeed*deltaTime

        x_error = self.x_raw - self.x_smooth
        y_error = self.y_raw - self.y_smooth

        if abs(x_error) < x_step:
            self.x_smooth = self.x_raw
        else:
            self.x_smooth += math.copysign(x_step,x_error)

        if abs(y_error) < y_step:
            self.y_smooth = self.y_raw
        else:
            self.y_smooth += math.copysign(y_step,y_error)

        print("[%f, %f]"%(self.x_smooth,self.y_smooth))

    def getSmooth(self):
        return (self.x_smooth, self.y_smooth)


class NeuralNetworkController:
    def __init__(self,imgLength):
        self.sess = tf.Session()

        self.imgPlaceholder = tf.placeholder(tf.float32,shape=[None,imgLength])




def main():
    sim = Simulation()
    inputSmoother = InputSmoother(0.5)
    pygame.init ()
    screenSurface = pygame.display.set_mode ((40, 40))
    fps = 30
    deltaTime = 1.0/fps
    clock = pygame.time.Clock()
    try:
        while True:


            #get the latest frame form the Simulation
            frame = sim.getFrame()

            #show the frame to the user
            pygame.surfarray.blit_array(screenSurface,np.transpose(frame*255))
            pygame.display.flip()

            #wait for the frame to finish
            clock.tick(fps)

            #show the frame to the nn

            #let pygame process the event queue
            pygame.event.pump()

            #get the state of all the keys
            keystate = pygame.key.get_pressed()

            inputSmoother.keyboardInput(keystate[pygame.K_UP],
                                        keystate[pygame.K_DOWN],
                                        keystate[pygame.K_LEFT],
                                        keystate[pygame.K_RIGHT])
            inputSmoother.update(deltaTime)
            x_input,y_input = inputSmoother.getSmooth()

            sim.input(x_input,y_input)
            sim.update(deltaTime)




    except KeyboardInterrupt:
        pass

    print("quiting")
    pygame.quit()

if __name__ == "__main__":
    main()
