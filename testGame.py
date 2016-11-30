# -*- coding:utf-8 -*-

from ctypes import *
import numpy as np
import time

IMAGE_ROW = 100
IMAGE_COL = 100
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
        self.hRenderBuffer = Objdll.createIntBuffer(IMAGE_ROW * IMAGE_COL)


    def getLeftOrRight(self):
        lOrR = Objdll.getLeftOrRight()
        return lOrR


    def moveBoard(self, action):
        Objdll.moveBoard(action)


    def getGrayImageByBuffer(self, channels):
        for i in range(IMAGE_ROW):
            for j in range(IMAGE_COL):
                color = Objdll.getValue(self.hRenderBuffer, i * IMAGE_ROW + j)
                if color == 0xff0000:
                    channels[i][j] = -0.5
                elif color == 0x00ff00:
                    channels[i][j] = 0.5
                else:
                    channels[i][j] = 0


    def render(self, w, h):
        Objdll.render(self.hRenderBuffer, w, h)


    def getNowImage(self):
        Objdll.getNowImage(self.hRenderBuffer)


    def updateGame(self, difficult):
        terminated = False
        tmp = Objdll.updateGame(difficult)
        if tmp == -1:
            terminated = False
        else:
            terminated = True
        time.sleep(0.016)
        return terminated



def __main__():
    state = gameState()
    buffer = state.createNewGame()
    while(Objdll.handleEvents()):
        action = state.getLeftOrRight()
        state.moveBoard(action)
        state.getNowImage()
        state.render(IMAGE_ROW, IMAGE_COL)
        state.updateGame(3)


if __name__ == "__main__":
    __main__()
