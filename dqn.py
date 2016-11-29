# -*- coding:utf-8 -*-
import numpy as np
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import testGame as tg

imgRow , imgCol = 80, 80
imgChannel = 2
actionNum = 3
initDistance = 1
batchSz = 32
gamma = 0.99
Initdifficult = 4

def getModel():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), init="normal", border_mode='same', input_shape=(imgChannel, imgRow, imgCol)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init="normal", border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init="normal", border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init="uniform"))
    model.add(Activation('relu'))
    model.add(Dense(actionNum, init="uniform"))

    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam)
    return model

def train(model):

    game = tg.gameState()    # 和迷宫当前状态相关的数据获取
    buffer = game.createNewGame()

    counter = 0    # 计数器
    difficult = Initdifficult
    while(True):

        inputs = np.zeros((batchSz, imgChannel, imgRow, imgCol))  # 32, 2, 80, 80
        targets = np.zeros((inputs.shape[0], actionNum))  # 32, 3

        for i in range(batchSz):
            state = game.getNowImage(buffer)
            inputs[i] = game.getTwoImageChannel(state) # 获取当前图像
            targets[i] = model.predict(inputs[i].reshape([1, imgChannel, imgRow, imgCol]))    # 网络走一步
            action_t = np.argmax(targets[i])    # reward预测值最大的那一步
            #print targets[i]
            # 按照网络预测走一步state.
            game.moveBoard(action_t)
            terminated = game.updateGame(difficult)

            if terminated:
                # 如果撞墙了或者走到终点了
                reward_t = -1
                targets[i][action_t] = reward_t

            else:
                # 要计算下一步reward
                state_t1 = game.getNowImage(buffer)
                image_t1 = game.getTwoImageChannel(state_t1)
                image_t1 = image_t1.reshape([1, imgChannel, imgRow, imgCol])
                targets[i][action_t] = 1.0 + gamma * np.max(model.predict(image_t1))
                #print action_t, targets[i]

        loss = model.train_on_batch(inputs, targets)

        if counter % 10 == 0:
            print "loss = %.4f" % loss

        if counter % 100 == 0:
            Flag = False
            while(True):

                test_state = game.getNowImage(buffer)
                image = game.getTwoImageChannel(test_state)
                game.render(buffer, imgRow, imgCol)
                test_action = np.argmax(model.predict(image.reshape([1, imgChannel, imgRow, imgCol])))
                game.moveBoard(test_action)
                Flag = game.updateGame(difficult)
                print Flag

                if Flag :
                    break

        if counter % 1000 == 0:
            # 保存一下权值
            model.save_weights("model.h5", overwrite=True)


        counter += 1    # 递增计数器


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
    



