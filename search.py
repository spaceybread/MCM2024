from alive_progress import alive_bar
import time
import numpy as np
import matplotlib.pyplot as plt
import random

rows, cols = (10000, 10000)
map = [[0 for i in range(cols)] for j in range(rows)]

for i in range(1000, 9001):
    map[9000][i] = 100
    
for i in range(1000, 5001):
    map[9000 - (2*(i - 1000))][i] = 100
    map[9000 - (2*(i - 1000))][10000 - i] = 100
    map[9000 - (2*(i - 1000) + 1)][i] = 100
    map[9000 - (2*(i - 1000) + 1)][10000 - i] = 100


def createWreck():
    y = random.randrange(1500, 9001)
    
    line = map[y]
    revline = line[::-1]
    
    l = line.index(100) + 1
    r = revline.index(100)
    
    x = random.randrange(r, l)
    
    saved = [y, x]
    map[saved[0]][saved[1]] = 5

def checkEveryBox():
    for yi in range(1500, 9001):
        line = map[yi]
        revline = line[::-1]
        l = line.index(100) + 1
        r = revline.index(100)
        
        for xi in range(r, l):
            if map[yi][xi] == 5:
                print("found", xi, yi, "base")
                #print(saved[0] == yi and saved[1] == xi)
                return True
    return False

def checkInRad(rad):
    for yi in range(1500, 9001):
        line = map[yi]
        revline = line[::-1]
        l = line.index(100) + 1
        r = revline.index(100)
        
        for xi in range(r, l, int(rad/20)):
            for r_1 in range(rad):
                for r_2 in range(rad):
                    if map[yi + r_1][xi + r_2] == 5:
                        print("found", xi, yi, "rad")
                        #print(saved[0] == yi + r_2 and saved[1] == xi + r_1)
                        return True
    return False
    
def sweep(rad):
    for xi in range(1000, 9001):
        for yi in range(1500, 9001, rad):
            for i in range(rad):
                if map[yi + i][xi] == 5:
                    print("found", xi, yi + i, "sweep")
                    return True
    return False
        
        

def clearMap():
    for y in range(10000):
        if 5 in map[y]:
            x = map[y].index(5)
            
            map[y][x] = 0
            print("clear")
            return


#TESTS = int(input("Run tests: "))
TESTS = 200
evBoxSum = 0
inRadSum = 0
sweepSum = 0

evBoxTimes = []
inRadTimes = []
sweepTimes = []

with alive_bar(TESTS) as bar:
    for test in range(TESTS):
        bar()
        createWreck()
        start = time.time()
        if checkEveryBox():
            evBoxSum += 1
        evBoxTimes.append(time.time() - start)
    
        start = time.time()
        if checkInRad(100):
            inRadSum += 1
        inRadTimes.append(time.time() - start)
        
        start = time.time()
        if sweep(200):
            sweepSum += 1
        sweepTimes.append(time.time() - start)
        
        clearMap()
        
print(evBoxSum, inRadSum, sweepSum)
print(min(evBoxTimes), min(inRadTimes), min(sweepTimes))
print(max(evBoxTimes), max(inRadTimes), max(sweepTimes))

#data = np.array(map)
#plt.imshow(data)
#plt.show()
