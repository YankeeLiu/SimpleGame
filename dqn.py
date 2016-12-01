# -*- coding:utf-8 -*-
import numpy as np
import time
from keras.models import Sequential
from keras.initializations import normal
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import testGame as tg
from collections import deque
from random import sample
#from copy import deepcopy
#from tqdm import tqdm

imgRow, imgCol = 100, 100
imgChannel = 4
actionNum = 3
initDistance = 1
batchSz = 64
gamma = 0.99
initDifficult = 4
observe = 3200
replayMemory = 20000

def getModel():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same', input_shape=(imgChannel, imgRow, imgCol)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(actionNum, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-5)
    model.compile(loss='mse', optimizer=adam)
    return model

class imgQueue:
    def __init__(self):
        self.__queue = deque()
        self.__info = deque()

    def append(self, img):
        if len(self.__queue) >= replayMemory:
           self.__queue.popleft()
        self.__queue.append(img)

    def addInfo(self, info):
        if len(self.__info) >= replayMemory - imgChannel:
           self.__info.popleft()
        self.__info.append(info)

    def getChannels(self, position=-1):
        grayImages_t = np.empty((1, imgChannel, imgRow, imgCol))

        position = len(self.__queue) - imgChannel if position == -1 else position

        for i in range(0, imgChannel):
            #get last 4 image
            grayImages_t[0][i] = self.__queue[i + position]

        return grayImages_t

    def getInfo(self, position=-1):
        return self.__info[position]

    def __len__(self):
        return len(self.__queue) - imgChannel + 1


def train(model):

    model.load_weights("model.h5")

    game = tg.gameState()    # 和当前状态相关的数据获取
    game.createNewGame()

    counter = -1    # 计数器
    difficult = initDifficult

    flag = False
    renderCounter = 0

    randomEpsilon = 0.0

    queueImg = imgQueue()

    for i in range(imgChannel):
        # Create continual 4 gray image
        game.getNowImage()
        queueImg.append(game.getGrayImageByBuffer())  # 获取当前图像
        game.updateGame(difficult)

    inputs = np.zeros((batchSz, imgChannel, imgRow, imgCol))
    targets = np.zeros((batchSz, actionNum))

    while(True):

        counter += 1

        if np.random.random() < randomEpsilon:
            action_t = np.random.randint(0, 3)
        else:
            grayImages_t = queueImg.getChannels()
            predict_action = model.predict(grayImages_t)
            action_t = np.argmax(predict_action)  # reward预测值最大的那一步

        game.moveBoard(action_t)
        terminated = game.updateGame(difficult)
        game.getNowImage()

        queueImg.append(game.getGrayImageByBuffer())  # 追加最新图像

        if terminated:
            # 如果撞了
            reward_t = -1
        else:
            # 要计算下一步reward
            grayImages_t = queueImg.getChannels()
            reward_t = 0.01 + gamma * np.max(model.predict(grayImages_t))

        queueImg.addInfo((action_t, reward_t))

        if counter < observe:
            continue
        #============TRAIN============

        for i in range(batchSz):
            choise = np.random.randint(0, len(queueImg) - 1)
            inputs[i] = queueImg.getChannels(choise)
            info = queueImg.getInfo(choise)
            action_history = info[0]
            reward_history = info[1]
            targets[i] = model.predict(inputs[i].reshape((1, imgChannel, imgRow, imgCol)))
            targets[i][action_history] = reward_history

        loss = model.train_on_batch(inputs, targets)

        if counter % 10 == 0:
            print "loss = %.4f" % loss

        if counter % 300 == 0:
            flag = True

        if(flag):
            game.render(imgRow, imgCol)
            # renderCounter += 1
            # if(renderCounter > 100):
            #     flag = False
            #     renderCounter = 0


        if counter % 1000 == 0:
            # 保存一下权值
            model.save_weights("model.h5", overwrite=True)

        randomEpsilon *= 0.99998


def __main__():
    model = getModel()
    train(model)
    # testRewardMatrix()


if __name__ == "__main__":
    __main__()
    



