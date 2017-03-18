import pygame as pg
import numpy as np
import theano
import theano.tensor as T
from collections import Counter
import random
import sys
from rpsconstants import *
rng = np.random


class RPS:
    def __init__(self, start, extra, reg, init_step, add_step, h_size, startlrn = 1000000000, multlrn = .5, endlrn = 150, insure_rate = 1 - 1e-5):
        pg.init()
        self.clock = pg.time.Clock()
        if pg.font:
            self.font = pg.font.Font(None,30)
        else:
            self.font = None
        self.screen = pg.display.set_mode(SCREEN_SIZE)
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
        self.h_size = h_size

        self.choice_ls = ('r', 'p', 's')
        self.choice_dict = {'r':0, 'p':1, 's':2}
        self.win_dict = {(0,0):0, (0,1):-1, (0,2):1, (1,0):1, (1,1):0, (1,2):-1, (2,0):-1, (2,1):1, (2,2):0}
        self.num_classes = 3 #Number of choices computer can make
        self.unique_features = 2 #Number of feature types that are unique and not simply the same as other features but for a different round
        self.vectordata = np.zeros(0)
        self.ans = np.zeros(0, dtype = 'int64')
        self.accuracy = []
        self.start_message_time = 200
        self.start_time = 0

        self.data = np.zeros((0,self.inputsize*2+self.unique_features))
        #Initialize Weights and Biases
        self.weights1 = rng.randn(np.size(self.data, axis=1),self.h_size)
        self.biases1 = rng.randn(self.h_size)
        self.weights2 = rng.randn(self.h_size,self.num_classes)
        self.biases2 = rng.randn(self.num_classes)

        self.init_game()
    def leftpos(self, location, totlocations):
        XMARGIN + (SCREEN_SIZE[0]-2*XMARGIN)*location/totlocations
    def toppos(self):
        YMARGIN + (SCREEN_SIZE[1]-2*YMARGIN)*2/3
    def buttlen(self,totlocationsx):
        w = (SCREEN_SIZE[0]-2*XMARGIN)*1/totlocationsx
        h = (SCREEN_SIZE[1]-2*YMARGIN)*1/3
        return w,h
    def init_game(self):
        self.fps = 50
        self.score = 0
        self.state = STATE_GUESSING
        self.start_message = "Hello!"
        width, height = self.buttlen()
        self.rock   = pg.Rect(leftpos(0,3),toppos(),width,height)
        self.paper  = pg.Rect(leftpos(1,3),toppos(),width,height)
        self.scissors = pg.Rect(leftpos(2,3), toppos(), width, height)
        self.buttons = [self.rock, self.paper, self.scissors]
        self.button_cols_init = [BLUE, BLUE, BLUE]
        self.button_cols = self.button_cols_init
        self.display = pg.Rect(SCREEN_SIZE[0]/2-DISP_SIZE[0]/2, SCREEN_SIZE[1]/2-DISP_SIZE[0]/2, DISP_SIZE[0], DISP_SIZE[1])
        self.disp_col = BRICK_COLOR
        self.banner   = pg.Rect(0, SCREEN_SIZE[1]/2 - BANNER_HEIGHT/2, SCREEN_SIZE[0], BANNER_HEIGHT)

    def draw_buttons(self):
        i = 0
        for button in self.buttons:
            pg.draw.rect(self.screen, self.button_cols[i], button)
            i+=1

    def draw_display(self):
        pg.draw.display(self.screen, self.disp_col, self.display)

    def show_message(self, message):
        if self.font:
            size = self.font.size(message) #use message font size
            font_surface = self.font.render(message,False, TEXT) #make font surface
            # Put font in center of screen
            x = (SCREEN_SIZE[0] - size[0]) / 2
            y = (SCREEN_SIZE[1] - size[1]) / 2
            pygame.draw.rect(self.screen, BLACK, self.banner) #place banner on screen
            self.screen.blit(font_surface, (x,y)) #Place on screen

    def check_mouse(self):
        self.button_cols = self.button_cols_init
        self.mousex, self.mousey = pg.mouse.get_pos()

    def update_colors(self):
        i=0
        for button in self.buttons:
            if button.collidepoint(self.mousex,self.mousey)
                self.button_cols[i] = MOUSEOVER
            i+=1

    def show_stats(self, r): #Show states at top of screen
        if self.font:
            font_surface = self.font.render("SCORE: " + str(self.score) + " ROUND: " + str(r), False, WHITE) #Specify text
            self.screen.blit(font_surface, (205,5)) #put on screen

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
        self.w1 = theano.shared(self.weights1, name="w1")#w = theano.shared(self.weights, name="w")
        self.w2 = theano.shared(self.weights2, name="w2")
        self.pseudo_w1 = theano.shared(self.weights1, name="psuedo_w1")#pseudo_w = theano.shared(self.weights, name="psuedo_w")
        self.pseudo_w2 = theano.shared(self.weights2, name="psuedo_w2")

        #Initialize bias vector b
        self.b1 = theano.shared(self.biases1, name="b1")
        self.b2 = theano.shared(self.biases2, name="b2")
        self.pseudo_b1 = theano.shared(self.biases1, name="pseudo_b1")
        self.pseudo_b2 = theano.shared(self.biases2, name="pseudo_b2")

        #construct formulas with Symbolics
        h1 = T.dot(x,self.w1)+self.b1
        h2 = T.dot(h1,self.w2)+self.b2
        sigma = T.nnet.softmax(h2) #SoftMax of the classes
        prediction = T.argmax(sigma, axis=1) #The most probable class.
        xent = -T.mean(T.log(sigma)[T.arange(y.shape[0]),y])
        cost = xent.mean() + self.reg * ((self.w1**2).sum() + (self.w2**2).sum()) #cost function with regularization
        gw1, gb1 = T.grad(cost, [self.w1,self.b1]) #Gradient computation
        gw2, gb2 = T.grad(cost, [self.w2,self.b2])

        #pseudo version for testing learning rates
        pseudo_h1 = T.dot(x,self.pseudo_w1)+self.pseudo_b1 #SoftMax of the classes
        pseudo_h2 = T.dot(pseudo_h1,self.pseudo_w2)+self.pseudo_b2
        pseudo_sigma = T.nnet.softmax(pseudo_h2)
        pseudo_prediction = T.argmax(pseudo_sigma, axis=1) #The most probable class.
        #xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
        pseudo_xent = -T.mean(T.log(pseudo_sigma)[T.arange(y.shape[0]),y])
        pseudo_cost = pseudo_xent.mean() + self.reg * ((self.pseudo_w1**2).sum() + (self.pseudo_w2**2).sum()) #cost function with regularization

        #Instantiate formulas with theano.function
        self.train = theano.function(inputs = [x,y],
                                outputs = [cost,gw1,gb1,gw2,gb2])
                                #updates = ((w, w- self.lrnrate*gw), (b, b- self.lrnrate*gb))) #Final Training
        self.pseudo_train = theano.function(inputs = [x,y],
                                outputs = [cost,gw1,gb1,gw2,gb2])
                                #updates = ((pseudo_w, w- self.lrnrate*gw), (pseudo_b, b- self.lrnrate*gb))) #Test cost if
        self.pseudo_cost = theano.function(inputs = [x,y],
                                      outputs = [pseudo_cost])
        self.predict_function = theano.function(inputs = [x], outputs = prediction)


        prevcost = sys.float_info.max
        finish = False
        traindata = self.data[:-1,:]
        for i in range(self.init_steps):
            print('trainstep: ' + str(i))
            self.lrnrate = self.startlrn
            for j in range(self.endlrn):

                cost,gw1,gb1,gw2,gb2 = self.pseudo_train(traindata,self.ans)
                self.pseudo_w1.set_value(self.w1.get_value()- self.lrnrate*gw1) #Manual Updates
                self.pseudo_b1.set_value(self.b1.get_value()- self.lrnrate*gb1) #Manual Updates
                self.pseudo_w2.set_value(self.w2.get_value()- self.lrnrate*gw2) #Manual Updates
                self.pseudo_b2.set_value(self.b2.get_value()- self.lrnrate*gb2) #Manual Updates

                test_cost = self.pseudo_cost(traindata,self.ans)[0]

                if test_cost < cost * self.insure_rate:
                    cost,gw1,gb1,gw2,gb2 = self.train(traindata,self.ans)
                    self.w1.set_value(self.w1.get_value()- self.lrnrate*gw1) #Manual Updates
                    self.b1.set_value(self.b1.get_value()- self.lrnrate*gb1) #Manual Updates
                    self.w2.set_value(self.w2.get_value()- self.lrnrate*gw2) #Manual Updates
                    self.b2.set_value(self.b2.get_value()- self.lrnrate*gb2) #Manual Updates
                    break
                if j == self.endlrn-1:
                    finish = True
                    break
                self.lrnrate *= self.multlrn
            if finish == True:
                break
            print('cost: ' + str(cost))
            #prevcost = cost
        self.weights1 = self.w1.get_value()
        self.biases1 = self.b1.get_value()
        self.weights2 = self.w2.get_value()
        self.biases2 = self.b2.get_value()
        #pred = predict(self.data[-1:,:])[0]
        #return pred
    def addNetwork(self):
        prevcost = sys.float_info.max
        finish = False
        traindata = self.data[:-1,:]
        for i in range(self.add_steps):
            print('trainstep: ' + str(i))
            self.lrnrate = self.startlrn
            for j in range(self.endlrn):

                cost,gw1,gb1,gw2,gb2 = self.pseudo_train(traindata,self.ans)
                self.pseudo_w1.set_value(self.w1.get_value()- self.lrnrate*gw1) #Manual Updates
                self.pseudo_b1.set_value(self.b1.get_value()- self.lrnrate*gb1) #Manual Updates
                self.pseudo_w2.set_value(self.w2.get_value()- self.lrnrate*gw2) #Manual Updates
                self.pseudo_b2.set_value(self.b2.get_value()- self.lrnrate*gb2) #Manual Updates

                test_cost = self.pseudo_cost(traindata,self.ans)[0]

                if test_cost < cost * self.insure_rate:
                    cost,gw1,gb1,gw2,gb2 = self.train(traindata,self.ans)
                    self.w1.set_value(self.w1.get_value()- self.lrnrate*gw1) #Manual Updates
                    self.b1.set_value(self.b1.get_value()- self.lrnrate*gb1) #Manual Updates
                    self.w2.set_value(self.w2.get_value()- self.lrnrate*gw2) #Manual Updates
                    self.b2.set_value(self.b2.get_value()- self.lrnrate*gb2) #Manual Updates
                    break
                if j == self.endlrn-1:
                    finish = True
                    break
                self.lrnrate *= self.multlrn
            if finish == True:
                break
            print('cost: ' + str(cost))
            #prevcost = cost
        self.weights1 = self.w1.get_value()
        self.biases1 = self.b1.get_value()
        self.weights2 = self.w2.get_value()
        self.biases2 = self.b2.get_value()

    def resetNetwork(self):
        self.weights1 = rng.randn(np.size(self.data, axis=1),self.h_size)
        self.biases1 = rng.randn(self.h_size)
        self.weights2 = rng.randn(self.h_size,self.num_classes)
        self.biases2 = rng.randn(self.num_classes)


        self.w1 = theano.shared(self.weights1, name="w1")#w = theano.shared(self.weights, name="w")
        self.w2 = theano.shared(self.weights2, name="w2")

        #Initialize bias vector b
        self.b1 = theano.shared(self.biases1, name="b1")
        self.b2 = theano.shared(self.biases2, name="b2")

        #Initialize Psuedo Tensors
        self.pseudo_w1 = theano.shared(self.weights1, name="psuedo_w1")#pseudo_w = theano.shared(self.weights, name="psuedo_w")
        self.pseudo_w2 = theano.shared(self.weights2, name="psuedo_w2")

        self.pseudo_b1 = theano.shared(self.biases1, name="pseudo_b1")
        self.pseudo_b2 = theano.shared(self.biases2, name="pseudo_b2")

    def rpsPredict(self):
        pred = self.predict_function(self.data[-1:,:])[0]
        return pred

    def get_inputs(self):
        while True:
            s = input('rock paper scissors (r|p|s):\n')
            if s == 'w':
                print(self.weights1)
                print(self.weights2)
            if not s or not s[0] in self.choice_ls:
                print('invalid choice')
                continue
            return s
    def pre_execute(self): #Make actual game update loop
        print("let's play a couple practice rounds")
        i= 0
        while i < (self.inputsize):
            print('round %d/%d :' % (i+1, self.inputsize))
            s = self.get_inputs()
            guessStr = random.choice(self.choice_ls)
            ansStr = s[0] #first letter of input
            guess = self.choice_dict[guessStr]
            play = self.choice_dict[ansStr]
            self.vectordata = np.append(self.vectordata, play)
            self.vectordata = np.append(self.vectordata, guess)
            print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,play)))
            #self.data[:,1]
            #print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,ans)))
            i += 1
    def extra_execute(self):
        print("\n\nJust a few more rounds!\n")
        i= 0
        while i < (self.extra):
            self.show_message('round %d/%d :' % (i+1, self.extra))
            s = self.get_inputs()
            guessStr = random.choice(self.choice_ls)
            ansStr = s[0] #first letter of input
            guess = self.choice_dict[guessStr]
            play = self.choice_dict[ansStr]
            self.vectordata = np.append(self.vectordata, play)
            self.vectordata = np.append(self.vectordata, guess)
            self.data = np.vstack((self.data, np.transpose(self.vectordata[i*self.unique_features:]))) # -self.unique_features#*(i+self.inputsize)
            print(self.data)
            if i != 0:
                self.ans = np.append(self.ans, self.correctChoice(play)) #start creating list of answers
                print(self.ans)
            print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,play)))
            #self.data[:,1]
            #print(guessStr + ', ' + ansStr + ', ' + str(self.evaluate(guess,ans)))
            i += 1
    def execute(self):
        print("\n\nNow for the real rounds!\n")
        i = self.extra
        while True:  #while i < 25:
            print('round %d :' % (i+1))
            s = self.get_inputs()
            #guessStr = random.choice(self.choice_ls)
            ansStr = s[0] #first letter of input
            play = self.choice_dict[ansStr]

            #Initialize Network
            if i is self.extra:
                self.initNetwork() #Initialize Network
            #Prediction

            guess = self.rpsPredict()

            self.vectordata = np.append(self.vectordata, play)
            self.vectordata = np.append(self.vectordata, guess)
            self.data = np.vstack((self.data, np.transpose(self.vectordata[i*self.unique_features:]))) # -self.unique_features #*(i+self.inputsize)

            if i != 0:
                self.ans = np.append(self.ans, self.correctChoice(play)) #start creating list of answers

            self.addNetwork() #Add to Network

            res = self.evaluate(guess,play)
            self.accuracy.append(res)
            print(self.choice_ls[guess] + ', ' + ansStr + ', ' + str(res))
            acc = Counter(self.accuracy)
            print(acc)
            #print("Accuracy: " + str(acc[1]/len(self.accuracy)))
            #print("Win Balance:" + str(sum(self.accuracy)))
            if i > self.inputsize*4 and Counter(self.accuracy[-self.inputsize*2:])[-1] > self.inputsize*3/2:
                print('performing poorly, resetting network')
                self.resetNetwork()
                i=0
                continue
            i += 1
    def run(self):
        game.pre_execute()
        game.extra_execute()
        game.execute()


if __name__ == '__main__':
    game = RPS(5, 5, 0, 500, 100, 6, insure_rate=1-1e-5) #, 100000, 1/3, 15, 1-1e-5)
    game.run()
