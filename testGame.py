# -*- coding:utf-8 -*-

from ctypes import *
import numpy as np
import time

IMAGE_ROW = 80
IMAGE_COL = 80
BOARD_WEIGHT = 15
BOARD_HIGH = 4
RENDER_SCALE = 4
NB_CHANNELS = 2

Objdll = CDLL("./game.so")

class gameState():

    state = np.zeros([IMAGE_ROW, IMAGE_COL])

    def __init__(self):
        pass

    def createNewGame(self):
        Objdll.initGame(IMAGE_ROW, IMAGE_COL, BOARD_WEIGHT, BOARD_HIGH)
        Objdll.initRenderer(IMAGE_ROW * RENDER_SCALE, IMAGE_COL * RENDER_SCALE, 32, RENDER_SCALE)
        hRenderBuffer = Objdll.createIntBuffer(IMAGE_ROW * IMAGE_COL)
        return hRenderBuffer

    def getLeftOrRight(self):
        lOrR = Objdll.getLeftOrRight()
        return lOrR

    def moveBoard(self, action):
        Objdll.moveBoard(action)

    def getNowImage(self, buffer):
        Objdll.getNowImage(buffer)
        for i in range(IMAGE_ROW):
            for j in range(IMAGE_COL):
                self.state[i][j] = Objdll.getValue(buffer, i*IMAGE_ROW + j)
                # print self.state[i][j]
        return self.state

    def getTwoImageChannel(self, state):
        channels = np.zeros([NB_CHANNELS, IMAGE_ROW, IMAGE_COL])
        for i in range(IMAGE_ROW):
            for j in range(IMAGE_COL):
                if state[i][j] == 0xff0000:
                    channels[0][i][j] = 1
                else:
                    channels[1][i][j] = -1
        return channels


    def render(self, buffer, w, h):
        Objdll.render(buffer, w, h)

    def updateGame(self, difficult):
        terminated = False
        tmp = Objdll.updateGame(difficult)
        if tmp == -1:
            terminated = False
        else:
            terminated = True
    # time.sleep(0.016)
        return terminated



def __main__():
    state = gameState()
    buffer = state.createNewGame()
    while(Objdll.handleEvents()):
        action = state.getLeftOrRight()
        state.moveBoard(action)
        state.getNowImage(buffer)
        state.render(buffer, IMAGE_ROW, IMAGE_COL)
        state.updateGame(3)


if __name__ == "__main__":
    __main__()
