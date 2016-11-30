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
from copy import deepcopy
#from tqdm import tqdm

imgRow , imgCol = 100, 100
imgChannel = 4
actionNum = 3
initDistance = 1
batchSz = 32
gamma = 0.99
initDifficult = 4
randomEpsilon = 0.1
observe = 320
replayMemory = 50000

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

def train(model):

    game = tg.gameState()    # 和当前状态相关的数据获取
    game.createNewGame()

    counter = -1    # 计数器
    difficult = initDifficult

    flag = False
    renderCounter = 0

    queue = deque()

    grayImages_t = np.empty((1, imgChannel, imgRow, imgCol))  # 1, 4, 100, 100

    for i in range(4):
        # Create continual 4 gray image
        game.getNowImage()
        game.getGrayImageByBuffer(grayImages_t[0][i])  # 获取当前图像
        game.updateGame(difficult)

    inputs = np.zeros((batchSz, imgChannel, imgRow, imgCol))
    targets = np.zeros((batchSz, actionNum))

    while(True):

        counter += 1

        if np.random.random() < randomEpsilon:
            action_t = np.random.randint(0, 3)
        else:
            predict_action = model.predict(grayImages_t)
            action_t = np.argmax(predict_action)  # reward预测值最大的那一步

        game.moveBoard(action_t)
        terminated = game.updateGame(difficult)

        grayImages_new = np.empty((1, imgChannel, imgRow, imgCol))
        for i in range(3):
            grayImages_new[0][i] = grayImages_t[0][i + 1]

        game.getNowImage()
        game.getGrayImageByBuffer(grayImages_new[0][3])  # 追加最新图像

        if terminated:
            # 如果撞了
            reward_t = -1
        else:
            # 要计算下一步reward
            reward_t = 0.01 + gamma * np.max(model.predict(grayImages_new))

        if len(queue) > replayMemory:
            queue.popleft()

        queue.append((grayImages_new, action_t, reward_t)) #image, action, reward

        if counter < observe:
            continue
        #============TRAIN============

        miniBatch = sample(queue, batchSz)

        for i in range(batchSz):
            inputs[i] = miniBatch[i][0]
            action_history = miniBatch[i][1]
            reward_history = miniBatch[i][2]
            targets[i] = model.predict(miniBatch[i][0])
            targets[i][action_history] = reward_history

        loss = model.train_on_batch(inputs, targets)

        if counter % 10 == 0:
            print "loss = %.4f" % loss

        if counter % 300 == 0:
            flag = True

        if(flag):
            game.render(imgRow, imgCol)
            renderCounter += 1
            if(renderCounter > 100):
                flag = False
                renderCounter = 0


        if counter % 1000 == 0:
            # 保存一下权值
            model.save_weights("model.h5", overwrite=True)


def __main__():
    model = getModel()
    train(model)
    # testRewardMatrix()


if __name__ == "__main__":
    __main__()
    



