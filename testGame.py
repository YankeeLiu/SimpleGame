# -*- coding:utf-8 -*-

from ctypes import *
import numpy as np
import time

IMAGE_ROW = 80
IMAGE_COL = 80
BOARD_WEIGHT = 15
BOARD_HIGH = 4
RENDER_SCALE = 4

Objdll = CDLL("./game.so")

class gameState():

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

    def render(self, buffer, w, h):
        Objdll.render(buffer, w, h)

    def updateGame(self, difficult):
        Objdll.updateGame(difficult)
        time.sleep(0.016)


state = gameState()
buffer = state.createNewGame()
while(Objdll.handleEvents()):
    action = state.getLeftOrRight()
    state.moveBoard(action)
    state.getNowImage(buffer)
    state.render(buffer, IMAGE_ROW, IMAGE_COL)
    state.updateGame(4)
