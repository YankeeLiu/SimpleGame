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
initDifficult = 3
randomEpsilon = 0.1
observe = 3200
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

    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    return model

def train(model):

    game = tg.gameState()    # 和当前状态相关的数据获取
    buffer = game.createNewGame()

    counter = -1    # 计数器
    difficult = initDifficult

    flag = False
    renderCounter = 0

    queue = deque()

    grayImages_t = np.empty((1, imgChannel, imgRow, imgCol))  # 4, 100, 100

    for i in range(4):
        # Create continue 4 gray image
        state = game.getNowImage(buffer)
        game.getGrayImageChannel(state, grayImages_t[0][i])  # 获取当前图像
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

        state = game.getNowImage(buffer)
        grayImages_new = np.empty((1, imgChannel, imgRow, imgCol))
        for i in range(3):
            grayImages_new[0][i] = deepcopy(grayImages_t[0][i + 1])
        game.getGrayImageChannel(state, grayImages_new[0][3])  # 追加最新图像

        if terminated:
            # 如果撞了
            reward_t = -1
        else:
            # 要计算下一步reward
            reward_t = 0.1 + gamma * np.max(model.predict(grayImages_new))

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

        if counter % 200 == 0:
            flag = True

        if(flag):
            game.render(buffer, imgRow, imgCol)
            renderCounter += 1
            if(renderCounter > 100):
                flag = False
                renderCounter = 0


        if counter % 1000 == 0:
            # 保存一下权值
            model.save_weights("model.h5", overwrite=True)



def testRewardMatrix():
    maze = MazeGame.mazeState()
    maze.createNewMaze()
    s_x, s_y, e_x, e_y = maze.selectStartAndEndPoint(maxDistance=10)
    reward = maze.calRewardMatrix(e_x, e_y)
    while(True):

        state = maze.getCurrentImage(s_x, s_y, e_x, e_y)
        maze.visualization(state)
        up_r = reward[s_x+1][s_y]
        down_r = reward[s_x-1][s_y]
        left_r = reward[s_x][s_y-1]
        right_r = reward[s_x][s_y+1]
        action = np.argmax([up_r, down_r, left_r, right_r])

        if action == 0:
            s_x += 1
        elif action == 1:
            s_x -= 1
        elif action == 2:
            s_y -= 1
        else:
            s_y += 1

        terminated, reward_r = maze.getReward(reward, s_x, s_y)
        print terminated, reward_r

        if terminated ==True:
            break

def __main__():
    model = getModel()
    train(model)
    # testRewardMatrix()


if __name__ == "__main__":
    __main__()
    



