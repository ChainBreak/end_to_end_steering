#!/usr/bin/env python

import numpy as np
import cv2
import pygame
import math
import tensorflow as tf

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
        self.maxTurnRate = 45

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

        #First Layer
        self.W_conv1 = self.weight_variable([5,5,3,36])
        self.b_conv1 = self.bias_variable([36])

        self.h_conv1 = tf.nn.relu( self.conv2d(self.x, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        # #Second Layer
        self.W_conv2 = self.weight_variable([3,3,36,16])
        self.b_conv2 = self.bias_variable([16])
        #
        self.h_conv2 = tf.nn.relu( self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        #self.h_conv2 =  self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2

        self.h_pool2 = self.max_pool_2x2(self.h_conv2)
        self.h_pool2 = self.blur_pool(self.h_pool2)
        self.h_pool2 = self.blur_pool(self.h_pool2)

        imgLength = 10*10*16
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, imgLength])

        #third layer fully Connected
        self.W_fc1 = self.weight_variable([imgLength,400])
        self.b_fc1 = self.bias_variable([400])

        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        #Fourth Layer Fully Connected Readout
        #self.W_fc2 = self.weight_variable([16*16*32,1])
        self.W_fc2 = self.weight_variable([400,1])
        self.b_fc2 = self.bias_variable([1])



        self.y_conv = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

        #self.y = tf.Print(self.y,[self.y],"Y: ")
        #y_conv = tf.Print(self.y_conv,[self.y_conv],"y_conv",summarize=5)
        sub = tf.subtract(self.y_,self.y_conv)

        #sub = tf.Print(sub,[sub],"subtract: ",summarize=5)
        self.dist = tf.sqrt( tf.reduce_sum( tf.square( sub), reduction_indices=1))

        #self.dist = tf.Print(self.dist, [self.dist], "Dist: ",summarize=5)

        self.loss = tf.reduce_mean(self.dist)
        #self.loss = tf.Print(self.loss,[self.loss],"Loss")
        self.train_step = tf.train.AdamOptimizer(0.00001).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.2, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x,W):
        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='VALID')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    def blur_pool(self,x):
        return tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID')

    def getAction(self,img):
        img = img.reshape(-1,64,64,3)
        return self.sess.run([self.y_conv, self.h_conv1, self.h_conv2, self.h_pool2],feed_dict={self.x: img})



    def train(self,frameTensor,actionTensor):
        return self.train_step.run(feed_dict={self.x: frameTensor, self.y_:actionTensor})

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



class FrameActionQueue():
    def __init__(self,maxLength):
        self.frameQueue = []
        self.actionQueue = []
        self.maxLength = maxLength

    def append(self,frameMat,actionMat):
        self.frameQueue.append(np.array([frameMat]))
        self.actionQueue.append(np.array([actionMat]))

        while len(self.frameQueue)>self.maxLength:
            del self.frameQueue[0]
            del self.actionQueue[0]

    def getFrameTensor(self):
        return np.concatenate(self.frameQueue,axis=0)

    def getActionTensor(self):
        return  np.concatenate(self.actionQueue,axis=0)

def activation2surface(activations):
    if len(activations.shape) == 4:
        activations = activations[0,:,:,:]
    h,w,c = activations.shape
    gridSize = int(math.ceil(math.sqrt(c)))
    img = np.zeros([(h+1)*gridSize,(w+1)*gridSize])
    for row in range(gridSize):
        for col in range(gridSize):
            i = row*gridSize + col
            if i < c:
                img[row*(h+1):(row+1)*(h+1)-1, col*(w+1):(col+1)*(w+1)-1] = activations[:,:,i]

    wMax = max(abs(img.min()),abs(img.max()))
    if wMax > 0.00001:
        img = img/wMax
        img *= 255

    surface = pygame.Surface(img.shape[0:2])
    pygame.surfarray.blit_array(surface,np.transpose(img)*1.0)
    surface = surface.convert()
    return surface


def main():
    pygame.init()

    viewSize = (64,64)
    screenSurface = pygame.display.set_mode ((1600,400))

    sim = DrivingSimulator2D('track5.png',viewSize)
    inputSmoother = InputSmoother(1.0)
    nn = NeuralNetworkController(viewSize)
    faq = FrameActionQueue(30)

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
            screenSurface.blit(surface,(0,0))


            # W = nn.getWeights()
            # wMax = max(abs(W.min()),abs(W.max()))
            # if wMax > 0.0001:
            #     W = W/wMax
            #     W *= 127
            # W += 128
            #
            # surface = pygame.Surface(W.shape[0:2])
            # pygame.surfarray.blit_array(surface,np.transpose(W,(1,0,2))*1.0)
            # surface = surface.convert()
            # surface = pygame.transform.scale(surface,(400,400))
            # screenSurface.blit(surface,(400,0))
            pygame.display.flip()


            clock.tick(fps)

            pygame.event.pump()
            inputSmoother.update(deltaTime)

            #get the user control action for this frame
            x_input_user,y_input_user = inputSmoother.getSmooth()

            x_input = x_input_user*1.5


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
                #print(x_input_nn)
                #clip the output of the nn to between -1 and 1
                x_input_nn = np.clip(action[0],-1.0,1.0)

                #print("[%f, %f]"%(x_input_user,y_input_user))
                x_input += x_input_nn

                x_input = np.clip(x_input,-1.0,1.0)

                if abs(x_input_user) > 0.0001 :
                    faq.append(frameScaled,x_input)
                    frameTensor = faq.getFrameTensor()
                    actionTensor = faq.getActionTensor()
                    #print(actionTensor)
                    nn.train(frameTensor,actionTensor)


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

            sim.input(x_input)
            sim.update(deltaTime)





    except KeyboardInterrupt:
        pass

    print("\nQuiting")
    pygame.quit()


if __name__ == "__main__":
    main()
