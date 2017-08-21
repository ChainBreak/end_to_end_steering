#!/usr/bin/env python

import numpy as np
import cv2
import pygame
import math
import tensorflow as tf

class DrivingSimulator2D:

    def __init__(self,trackFileName,viewSize):
        self.bg_img = cv2.imread(trackFileName,cv2.IMREAD_GRAYSCALE)
        self.bg_h, self.bg_w =self.bg_img.shape[0:2]
        self.view_h, self.view_w = viewSize
        self.view_d = math.sqrt(self.view_h**2 + self.view_w**2)/2
        self.x = self.bg_w / 2
        self.y = self.bg_h / 2

        self.speed = 30 #pixels per second
        self.dir = 00 #degrees
        self.turnRate = 00 #degrees per second
        self.maxTurnRate = 90

    def input(self,x_input,y_input):
        self.turnRate = -self.maxTurnRate * x_input

    def update(self,deltaTime):
        self.dir += self.turnRate * deltaTime
        self.x += self.speed * math.cos(math.radians(self.dir)) * deltaTime
        self.y -= self.speed * math.sin(math.radians(self.dir)) * deltaTime

        #position limits
        self.x = min(max(self.view_d,self.x), self.bg_w - self.view_d)
        self.y = min(max(self.view_d,self.y), self.bg_h - self.view_d)


    def getViewFrame(self):
        M = cv2.getRotationMatrix2D((self.x,self.y),-self.dir+90,1.0)
        M[0,2] += self.view_w/2 - self.x
        M[1,2] += self.view_h/2 - self.y
        view = cv2.warpAffine(self.bg_img,M,(self.bg_h,self.bg_w))
        return view[0:self.view_h, 0:self.view_w].copy()

class InputSmoother:
    """up is positive, right is positive"""
    def __init__(self,timeToFullSpeed):
        self.timeToFullSpeed = timeToFullSpeed
        self.x_smooth = 0.0
        self.y_smooth = 0.0

    def update(self,deltaTime):
        x_step = 1.0/self.timeToFullSpeed*deltaTime
        y_step = 1.0/self.timeToFullSpeed*deltaTime

        keystate = pygame.key.get_pressed()
        x_raw = keystate[pygame.K_RIGHT] - keystate[pygame.K_LEFT]
        y_raw = keystate[pygame.K_UP] - keystate[pygame.K_DOWN]

        x_error = x_raw - self.x_smooth
        y_error = y_raw - self.y_smooth

        if abs(x_error) < x_step:
            self.x_smooth = x_raw
        else:
            self.x_smooth += math.copysign(x_step,x_error)

        if abs(y_error) < y_step:
            self.y_smooth = y_raw
        else:
            self.y_smooth += math.copysign(y_step,y_error)

        #print("[%f, %f]"%(self.x_smooth,self.y_smooth))

    def getSmooth(self):
        return (self.x_smooth, self.y_smooth)

class NeuralNetworkController:
    def __init__(self,imgSize):
        self.imgSize = imgSize

        self.imgLength = imgSize[0]*imgSize[1]
        self.sess = tf.InteractiveSession()

        self.x = tf.placeholder(tf.float32,shape=[None,self.imgLength])
        self.y_ = tf.placeholder(tf.float32,shape=[None,2])

        self.W = tf.Variable(tf.zeros([self.imgLength,2]))
        self.b = tf.Variable(tf.zeros([2]))

        self.sess.run(tf.global_variables_initializer())

        self.y = tf.matmul(self.x,self.W) + self.b

        #self.y = tf.Print(self.y,[self.y],"Y: ")

        sub = tf.subtract(self.y_,self.y)

        #sub = tf.Print(sub,[sub],"subtract: ")
        self.dist = tf.sqrt( tf.reduce_sum( tf.square( sub), reduction_indices=1))

        #self.dist = tf.Print(self.dist, [self.dist], "Dist: ")

        self.train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(self.dist)

    def getAction(self,img):
        img = img.reshape(-1,self.imgLength)
        action = self.y.eval(feed_dict={self.x: img})
        return (action[0,0],action[0,1])


    def train(self,img,x_input,y_input):
        img = img.reshape(-1,self.imgLength)
        correctAction = np.array([[x_input,y_input]])
        return self.train_step.run(feed_dict={self.x: img, self.y_:correctAction})

    def reinitialize(self):
        self.sess.run(tf.global_variables_initializer())

    def getWeights(self):
        W = self.sess.run(self.W)
        W = W.reshape(self.imgSize[0],self.imgSize[1],2)
        return W

def main():
    pygame.init()
    screenSurface = pygame.display.set_mode ((400, 400))
    viewSize = (60,60)

    sim = DrivingSimulator2D('track1.png',viewSize)
    inputSmoother = InputSmoother(1.0)
    nn = NeuralNetworkController(viewSize)

    fps = 30
    deltaTime =  1.0/fps
    clock = pygame.time.Clock()
    nnEnabled = True
    try:

        while True:



            #get the latest frame form the Simulation
            frame = sim.getViewFrame()
            #print(frame.shape)
            frameT = np.transpose(frame,(1,0))
            frameScaled = frame / 255
            #print(frame.shape)
            #show the frame to the user
            print(frameT.shape)
            surface = pygame.Surface(viewSize)
            pygame.surfarray.blit_array(surface,frameT.astype('uint8'))
            surface = surface.convert()
            surface = pygame.transform.scale(surface,(400,400))
            screenSurface.blit(surface,(0,0))
            pygame.display.flip()


            clock.tick(fps)


            pygame.event.pump()
            inputSmoother.update(deltaTime)

            #get the user control action for this frame
            x_input_user,y_input_user = inputSmoother.getSmooth()

            x_input = x_input_user
            y_input = y_input_user

            if nnEnabled:
                #get the control action from the nn for this frame
                x_input_nn,y_input_nn = nn.getAction(frameScaled)

                #clip the output of the nn to between -1 and 1
                x_input_nn = np.clip(x_input_nn,-1.0,1.0)
                y_input_nn = np.clip(y_input_nn,-1.0,1.0)

                #print("[%f, %f]"%(x_input_user,y_input_user))
                x_input += x_input_nn
                y_input += y_input_nn


                x_input = np.clip(x_input,-1.0,1.0)
                y_input = np.clip(y_input,-1.0,1.0)

                input_user = math.sqrt(x_input_user**2 + y_input_user**2)
                if input_user > 0.0001 :
                    nn.train(frameScaled,x_input,y_input)

            sim.input(x_input,y_input)
            sim.update(deltaTime)





    except KeyboardInterrupt:
        pass

    print("\nQuiting")
    pygame.quit()


if __name__ == "__main__":
    main()
