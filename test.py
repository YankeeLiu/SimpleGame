imgRow, imgCol = 2, 2
imgChannel = 4
observe = 3
replayMemory = 10

import numpy as np
from collections import deque

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



q = imgQueue()

for i in range(4):
    img = np.zeros((imgRow, imgCol))
    q.append(img)

for i in range(30):
    print "use", q.getChannels(), "to cal", i
    img = np.zeros((imgRow, imgCol)) + i + 1
    q.append(img)
    q.addInfo(i)

for i in range(5):
    print q.getChannels(i)
    print q.getInfo(i)