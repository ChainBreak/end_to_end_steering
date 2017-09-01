#!/usr/bin/env python

import numpy as np
import cv2
import pygame
import math
import tensorflow as tf
from randomWeightedRecal import RandomWeightedRecal

class DrivingSimulator2D:

    def __init__(self,trackFileName,viewSize):
        self.bg_img = cv2.imread(trackFileName)
        self.bg_img = self.bg_img[:,:,::-1] #convert BGR to RGB
        self.bg_h, self.bg_w =self.bg_img.shape[0:2]
        self.view_h, self.view_w = viewSize
        self.view_d = math.sqrt(self.view_h**2 + self.view_w**2)/2
        self.x = self.bg_w / 2
        self.y = self.bg_h / 2

        self.speed = 30 #pixels per second
        self.dir = 00 #degrees
        self.turnRate = 00 #degrees per second
        self.maxTurnRate = 90

    def input(self,x_input):
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
        M[1,2] += self.view_h - self.y
        view = cv2.warpAffine(self.bg_img,M,(self.bg_h,self.bg_w))
        return view[0:self.view_h, 0:self.view_w,:].copy()

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

        self.sess = tf.InteractiveSession()
        #[batches,height,width,chanels]
        self.x = tf.placeholder(tf.float32,shape=[None,64,64,3])
        self.y_ = tf.placeholder(tf.float32,shape=[None,1])
        self.keep_prob = tf.placeholder(tf.float32)

        #First Layer
        self.W_conv1 = self.weight_variable([5,5,3,24])
        self.b_conv1 = self.bias_variable([24])
        self.h_conv1 = self.leaky_relu( self.conv2d(self.x, self.W_conv1,2) + self.b_conv1)

        #Second Layer
        self.W_conv2 = self.weight_variable([5,5,24,36])
        self.b_conv2 = self.bias_variable([36])
        self.h_conv2 = self.leaky_relu( self.conv2d(self.h_conv1, self.W_conv2,2) + self.b_conv2)

        #Third Layer
        self.W_conv3 = self.weight_variable([5,5,36,48])
        self.b_conv3 = self.bias_variable([48])
        self.h_conv3 = self.leaky_relu( self.conv2d(self.h_conv2, self.W_conv3,2) + self.b_conv3)


        print(self.h_conv3.get_shape())
        n,h,w,c = self.h_conv3.get_shape()
        imgLength = int(h*w*c)
        self.h_flat = tf.reshape(self.h_conv3, [-1, imgLength])

        #third layer fully Connected
        fc1_size =  int(c) * 4
        self.W_fc1 = self.weight_variable([imgLength,fc1_size])
        self.b_fc1 = self.bias_variable([fc1_size])

        self.h_fc1 = self.leaky_relu(tf.matmul(self.h_flat, self.W_fc1) + self.b_fc1)

        #self.h_fc1 = tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1
        self.h_drop = tf.nn.dropout(self.h_fc1,self.keep_prob)
        #Fourth Layer Fully Connected Readout
        #self.W_fc2 = self.weight_variable([16*16*32,1])
        self.W_fc2 = self.weight_variable([fc1_size,1])
        self.b_fc2 = self.bias_variable([1])

        self.y_conv = tf.matmul(self.h_drop, self.W_fc2) #+ self.b_fc2

        #self.y = tf.Print(self.y,[self.y],"Y: ")
        #y_conv = tf.Print(self.y_conv,[self.y_conv],"y_conv",summarize=5)
        sub = tf.subtract(self.y_,self.y_conv)

        #sub = tf.Print(sub,[sub],"subtract: ",summarize=5)
        self.dist = tf.sqrt( tf.reduce_sum( tf.square( sub), reduction_indices=1))

        #self.dist = tf.Print(self.dist, [self.dist], "Dist: ",summarize=5)

        self.loss = tf.reduce_mean(self.dist)

        #self.loss = tf.Print(self.loss,[self.loss],"Loss")

        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.2, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x,W,s):
        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        return tf.nn.conv2d(x,W, strides=[1,s,s,1], padding='VALID')
    @staticmethod
    def leaky_relu(x):
        alpha = 0.001
        return tf.maximum(x,x*alpha)

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    def blur_pool(self,x):
        return tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID')

    def getAction(self,img):
        img = img.reshape(-1,64,64,3)
        return self.sess.run([self.y_conv, self.h_conv1, self.h_conv2, self.h_conv3],feed_dict={self.x: img, self.keep_prob:1.0})

    def train(self,frameTensor,actionTensor):
        return self.train_step.run(feed_dict={self.x: frameTensor, self.y_:actionTensor, self.keep_prob:1.0})

    def reinitialize(self):
        self.sess.run(tf.global_variables_initializer())

    def getWeights(self):
        W = self.sess.run(self.W_conv1)
        #[5,5,3,16]
        W_img = np.zeros([10*4,10*4,3])

        for row in range(4):
            for col in range(4):
                filterIndex = row*4 + col
                W_img[row*10+1:row*10+10,col*10+1:col*10+10,:] = W[:,:,:,filterIndex]

        return W_img




def activation2surface(activations):
    if len(activations.shape) == 4:
        activations = activations[0,:,:,:]

    activations -= activations.min()
    wMax = activations.max()
    if wMax > 0.00001:
        activations = activations/wMax
        activations *= 255

    h,w,c = activations.shape
    h += 1
    w += 1
    gridSize = int(math.ceil(math.sqrt(c)))
    img = np.zeros([(h+1)*gridSize,(w+1)*gridSize])
    for row in range(gridSize):
        for col in range(gridSize):
            i = row*gridSize + col
            if i < c:
                act = activations[:,:,i]

                img[row*h:(row+1)*h-1, col*w:(col+1)*w-1] = act


    surface = pygame.Surface(img.shape[0:2])
    pygame.surfarray.blit_array(surface,np.transpose(img)*1.0)
    surface = surface.convert()
    return surface


def main():
    pygame.init()

    viewSize = (64,64)
    screenSurface = pygame.display.set_mode ((1600,400))

    sim = DrivingSimulator2D('track8.png',viewSize)
    inputSmoother = InputSmoother(0.5)
    nn = NeuralNetworkController(viewSize)

    rr = RandomWeightedRecal(4000)

    fps = 30
    deltaTime =  1.0/fps
    clock = pygame.time.Clock()
    nnEnabled = False
    print("\n***Controls***\n\
arrow keys: velocity input\n\
         n: enable/disable neural network Controls\n\
         c: reinitialize neural netowork weights\n\
     space: random position\n")

    print("Neural Network %s" % ["Disabled","Enabled"][nnEnabled])
    try:

        while True:

            #get the latest frame form the Simulation
            frame = sim.getViewFrame()

            #show the frame to the user

            surface = pygame.Surface(viewSize)
            pygame.surfarray.blit_array(surface,np.transpose(frame,(1,0,2))*1.0)
            surface = surface.convert()
            surface = pygame.transform.scale(surface,(400,400))
            pygame.draw.line(surface,0xff0000,(200,200),(200,400))
            screenSurface.blit(surface,(0,0))


            pygame.display.flip()


            clock.tick(fps)

            pygame.event.pump()
            inputSmoother.update(deltaTime)

            #get the user control action for this frame
            x_input_user,y_input_user = inputSmoother.getSmooth()

            turn = x_input_user*2.0


            if nnEnabled:
                frameScaled = frame / 255.0
                #get the control action from the nn for this frame
                action,acts1,acts2,acts3 = nn.getAction(frameScaled)
                actSurf = activation2surface(acts1)
                actSurf = pygame.transform.scale(actSurf,(400,400))
                screenSurface.blit(actSurf,(400,0))

                actSurf = activation2surface(acts2)
                actSurf = pygame.transform.scale(actSurf,(400,400))
                screenSurface.blit(actSurf,(800,0))

                actSurf = activation2surface(acts3)
                actSurf = pygame.transform.scale(actSurf,(400,400))
                screenSurface.blit(actSurf,(1200,0))


                turn_nn = np.clip(float(action[0]),-1,1)

                keystate = pygame.key.get_pressed()
                focused = keystate[pygame.K_UP]
                if not focused:
                    turn += turn_nn

                turn = np.clip(turn,-1,1)


                if abs(x_input_user) > 0.0001 or focused:
                    correctAction = np.array([turn])
                    if len(rr.weightedItemList)%50 == 0:
                        print(len(rr.weightedItemList))
                    if len(rr.weightedItemList) > 40:
                        frameTensor = np.array([frameScaled])
                        actionTensor = np.array([correctAction])

                        frameActionList = rr.getWeightedRandomList(39)
                        for oldFrame, oldAction in frameActionList:
                            frameTensor = np.append(frameTensor, oldFrame,axis=0)
                            actionTensor = np.append(actionTensor, oldAction,axis=0)
                        #print(frameTensor.shape)
                        #print(actionTensor)
                        nn.train(frameTensor,actionTensor)

                    rr.addItem(abs(x_input_user)+float(focused),( np.array([frameScaled]), np.array([correctAction]) ))


            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    keystate = pygame.key.get_pressed()
                    if keystate[pygame.K_SPACE] :
                        sim.randomPosition()
                        print("Random Position")
                    elif keystate[pygame.K_c]:
                        nn.reinitialize()
                        print("Reinitialized Weights")
                    elif keystate[pygame.K_n]:
                        nnEnabled = not nnEnabled
                        print("Neural Network %s" % ["Disabled","Enabled"][nnEnabled])

            sim.input(turn)
            sim.update(deltaTime)





    except KeyboardInterrupt:
        pass

    print("\nQuiting")
    pygame.quit()


if __name__ == "__main__":
    main()
