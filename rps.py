import pygame as pg
import numpy as np
import theano
import theano.tensor as T
from collections import Counter
import random
import sys
rng = np.random


class RPS:
    def __init__(self, start, extra, reg, init_step, add_step, startlrn, multlrn, endlrn, insure_rate):
        self.choice_ls = ('r', 'p', 's')
        self.choice_dict = {'r':0, 'p':1, 's':2}
        self.win_dict = {(0,0):0, (0,1):-1, (0,2):1, (1,0):1, (1,1):0, (1,2):-1, (2,0):-1, (2,1):1, (2,2):0}
        self.data = np.zeros((0,start))
        self.vectordata = np.zeros((0,1))
        self.ans = np.zeros(0, dtype = 'int64')
        self.accuracy = []
        self.num_classes = 3 #Number of choices computer can make
        self.weights = rng.randn(np.size(self.data, axis=1),self.num_classes)
        self.inputsize = start
        self.extra = extra
        self.reg = reg
        self.init_steps = init_step #training steps
        self.add_steps = add_step
        self.startlrn = startlrn
        self.lrnrate = self.startlrn
        self.multlrn = multlrn
        self.endlrn = endlrn
        self.insure_rate = insure_rate
    def evaluate(self,com,play):
        gameres = (com,play)
        return self.win_dict[gameres]
    def correctChoice(self,op_choice):
        r = op_choice+1
        if r > 2:
            r = 0
        return r
    def initNetwork(self):
        #Declare Theano Symbolics
        x = T.dmatrix('x')
        y = T.lvector('y')

        #Initialize weight vector randomly (use shared so they stay the same after the generation)
        w = theano.shared(self.weights, name="w")
        pseudo_w = theano.shared(self.weights, name="psuedo_w")

        #Initialize bias vector b
        b = theano.shared(0.01, name="b")
        pseudo_b = theano.shared(0.01, name="pseudo_b")

        #construct formulas with Symbolics
        sigma = T.nnet.softmax(T.dot(x,w)+b) #SoftMax of the classes
        prediction = T.argmax(sigma, axis=1) #The most probable class.
        xent = -T.mean(T.log(sigma)[T.arange(y.shape[0]),y])
        cost = xent.mean() + self.reg * (w**2).sum() #cost function with regularization
        gw, gb = T.grad(cost, [w,b]) #Gradient computation

        #pseudo version for testing learning rates
        pseudo_sigma = T.nnet.softmax(T.dot(x,pseudo_w)+pseudo_b) #SoftMax of the classes
        pseudo_prediction = T.argmax(pseudo_sigma, axis=1) #The most probable class.
        #xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
        pseudo_xent = -T.mean(T.log(pseudo_sigma)[T.arange(y.shape[0]),y])
        pseudo_cost = pseudo_xent.mean() + self.reg * (pseudo_w**2).sum() #cost function with regularization

        #Instantiate formulas with theano.function
        train = theano.function(inputs = [x,y],
                                outputs = [cost,gw,gb])
                                #updates = ((w, w- self.lrnrate*gw), (b, b- self.lrnrate*gb))) #Final Training
        pseudo_train = theano.function(inputs = [x,y],
                                outputs = [cost])
                                #updates = ((pseudo_w, w- self.lrnrate*gw), (pseudo_b, b- self.lrnrate*gb))) #Test cost if
        pseudo_cost = theano.function(inputs = [x,y],
                                      outputs = [pseudo_cost])
        predict = theano.function(inputs = [x], outputs = prediction)


        prevcost = sys.float_info.max
        finish = False

        for i in range(self.init_steps):
            print('trainstep: ' + str(i))
            self.lrnrate = self.startlrn
            for j in range(self.endlrn):

                cost,gw,gb = pseudo_train(self.data,self.ans)
                pseudo_w.set_value(w.get_value()- self.lrnrate*gw) #Manual Updates
                pseudo_b.set_value(b.get_value()- self.lrnrate*gb) #Manual Updates

                pseudo_cost = pseudo_cost(self.data,self.ans)[0]

                if pseudo_cost < prevcost * self.insure_rate:
                    cost,gw,gb = train(self.data,self.ans)
                    w.set_value(w.get_value()- self.lrnrate*gw) #Manual Updates
                    b.set_value(b.get_value()- self.lrnrate*gb) #Manual Updates
                    break
                if j == self.endlrn-1:
                    finish = True
                    break
                self.lrnrate *= self.multlrn
            print('cost: ' + str(cost))
            if finish == True:
                break
            prevcost = cost
        self.weights = w.get_value()
        pred = predict(self.data[-1:,:])[0]
        return pred


    def addNetwork(self):
        print('blah')
    def get_inputs(self):
        while True:
            s = input('rock paper scissors (r|p|s):\n')
            if not s or not s[0] in self.choice_ls:
                print('invalid choice')
                continue
            return s
    def pre_execute(self):
        print("let's play a couple practice rounds")
        i= 0
        while i < (self.inputsize):
            print('round %d/%d :' % (i+1, self.inputsize))
            s = self.get_inputs()
            guessStr = random.choice(self.choice_ls)
            ansStr = s[0] #first letter of input
            guess = self.choice_dict[guessStr]
            play = self.choice_dict[ansStr]

            self.vectordata = np.vstack((self.vectordata, play))
            print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,play)))
            #self.data[:,1]
            #print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,ans)))
            i += 1
    def extra_execute(self):
        print("\n\nJust a few more rounds!\n")
        i= 0
        while i < (self.extra):
            print('round %d/%d :' % (i+1, self.extra))
            s = self.get_inputs()
            guessStr = random.choice(self.choice_ls)
            ansStr = s[0] #first letter of input
            guess = self.choice_dict[guessStr]
            play = self.choice_dict[ansStr]
            self.vectordata = np.vstack((self.vectordata, play))
            self.data = np.vstack((self.data, np.transpose(self.vectordata[i:i+self.inputsize,:])))
            self.ans = np.append(self.ans, self.correctChoice(play)) #start creating list of answers
            print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,play)))
            #self.data[:,1]
            #print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,ans)))
            i += 1
    def execute(self):
        print("\n\nNow for the real rounds!\n")
        i=0
        while True:  #while i < 25:
            print('round %d :' % (i+1+self.inputsize))
            s = self.get_inputs()
            #guessStr = random.choice(self.choice_ls)
            ansStr = s[0] #first letter of input
            play = self.choice_dict[ansStr]
            guess = self.initNetwork()
            self.vectordata = np.vstack((self.vectordata, play))
            self.data = np.vstack((self.data, np.transpose(self.vectordata[i:i+self.inputsize,:])))
            self.ans = np.append(self.ans, self.correctChoice(play)) #start creating list of answers
            res = self.evaluate(guess,play)
            self.accuracy.append(res)
            print(self.choice_ls[guess] + ', ' + ansStr + ', ' + str(res))
            acc = Counter(self.accuracy)
            print("Accuracy: " + str(acc[1]/len(self.accuracy)))
            print("Win Balance:" + str(sum(self.accuracy)))
            i += 1


if __name__ == '__main__':
    game = RPS(6, 3, 0, 100, 5, 100000, 1/3, 15, 1)
    game.pre_execute()
    game.extra_execute()
    game.execute()
