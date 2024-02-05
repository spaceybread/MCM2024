from alive_progress import alive_bar

def spot(x, y, map):
    map[x][y] = 10
    
    for i in range(9):
        if (x - i) > -1:
            map[x - i][y] = 10 - i
        if (x + i) < len(map):
            map[x + i][y] = 10 - i

    for i in range(9):
        if (y - i) > -1:
            map[x][y - i] = 10 - i
        if (y + i) < len(map):
            map[x][y + i] = 10 - i
    
    for i in range(6):
        if (y - i) > -1:
            if (x - i) > -1:
                map[x - i][y - i] = 10 - i
            if (x + i) < len(map):
                map[x + i][y - i] = 10 - i
        if (y + 1) < len(map):
            if (x - i) > -1:
                map[x - i][y + i] = 10 - i
            if (x + i) < len(map):
                map[x + i][y + i] = 10 - i
                
def spot2(x, y, map, r):
    with alive_bar(2400*2400) as bar:
        for i in range(len(map)):
            for j in range(len(map)):
                bar()
                if (x - i)**2 + (y - j)**2 < r**2:
                    if map[i][j] == 0:
                        map[i][j] = 1 - (((x - i)**2 + (y - j)**2)**(0.5))/r
                    else:
                        map[i][j] = (map[i][j] + (1 - (((x - i)**2 + (y - j)**2)**(0.5))/r))
    
    
import numpy as np
import matplotlib.pyplot as plt
import random

rows, cols = (2400, 2400)
map = [[0 for i in range(cols)] for j in range(rows)]

def randomGen():
    xs = []
    ys = []
    cords = []
    for i in range(10):
        print(i)
        x = 0
        while True:
            x = random.randrange(400)
        
            flag = False
        
            for val in xs:
                if abs(x - val) < 32:
                    flag = True
            #print(flag)
            if flag == False:
                xs.append(x)
                break
    
        while True:
            x = random.randrange(400)
        
            flag = False
            for val in ys:
                if abs(x - val) < 32:
                    flag = True
            #print(flag)
            if flag == False:
                ys.append(x)
                break
            
            
        spot2(xs[-1], ys[-1], map, 90)
        cords.append([xs[-1], ys[-1]])
    
    return cords
    
    #spot2(random.randrange(463), random.randrange(463), map, 80)

def spreadGen(r):
    cords = []
    x = [300, 900, 1500, 2100]
    
    for i in x:
        for j in x:
            cords.append([i, j])
            spot2(i, j, map, r)

    return cords
    
def spreadDiagonalGen(r):
    cords = []
    x = [300, 900, 1500, 2100]
    
    for i in x:
        for j in x:
            cords.append([i, j])
            spot2(i, j, map, r)
            
    x = [0, 600, 1200, 1800, 2400]
    for i in x:
        for j in x:
            cords.append([i, j])
            spot2(i, j, map, r)
    
    return cords

def spreadHexagon(r):
    cords = []
    xflat = [0, 600, 1200, 1800, 2400]
    xpoint = [900 - 580, 1500 - 580, 2100 - 580, 2700 - 580]
    
    #xflat = [0, 200, 400, 600, 800, 1000]
    #xpoint = [300 - 193, 700 - 193, 1100 - 193]
    
    for i in xpoint:
        cords.append([300 - 560, i])
        spot2(300 - 560, i, map, r)
    
    
    for i in xflat:
        cords.append([300, i])
        spot2(300, i, map, r)
    
    for i in xpoint:
        cords.append([300 + 560, i])
        spot2(300 + 560, i, map, r)
    
    for i in xflat:
        cords.append([300 + 560*2, i])
        spot2(300 + 560*2, i, map, r)

    for i in xpoint:
        cords.append([300 + 560*3, i])
        spot2(300 + 560*3, i, map, r)

    for i in xflat:
        cords.append([300 + 560*4, i])
        spot2(300 + 560*4, i, map, r)

    for i in xpoint:
        cords.append([300 + 560*5, i])
        spot2(300 + 560*5, i, map, r)
    return cords

def spreadHexRing(r):
    cords = []
    xflat = [0, 1200, 2400]
    xpoint = [900 - 580, 1500 - 580, 2100 - 580, 2700 - 580]
    
    #xflat = [0, 200, 400, 600, 800, 1000]
    #xpoint = [300 - 193, 700 - 193, 1100 - 193]
    
    for i in xflat:
        cords.append([300, i])
        spot2(300, i, map, r)
    
    for i in xpoint:
        cords.append([300 + 560, i])
        spot2(300 + 560, i, map, r)
    
    for i in xflat:
        cords.append([300 + 560*2, i])
        spot2(300 + 560*2, i, map, r)

    for i in xpoint:
        cords.append([300 + 560*3, i])
        spot2(300 + 560*3, i, map, r)

    for i in xflat:
        cords.append([300 + 560*4, i])
        spot2(300 + 560*4, i, map, r)

    for i in xpoint:
        cords.append([300 + 560*5, i])
        spot2(300 + 560*5, i, map, r)
    

    return cords



out = []
packer = int(input("Pack strat: "))
if packer == 0:
    out = randomGen()
elif packer == 1:
    out = spreadGen(350)
elif packer == 2:
    out = spreadDiagonalGen(350)
elif packer == 3:
    out = spreadHexagon(350)
elif packer == 4:
    out = spreadHexRing(350)



ranOut = []
#print(map)
sum = 0

with alive_bar(2400*2400) as bar:
    for i in range(len(map)):
        for j in range(len(map)):
            #print(i, j)
            bar()
            if map[i][j] != 0:
                sum += 1
            else:
                dist = []
                for g in out:
                    val = (abs((i - g[0])**2 + (j - g[1])**2))**(0.5)
                    dist.append(val)
                ranOut.append(min(dist))
            
            
coverage = round(sum/(rows * cols), 2)

maxRange = 0
if len(ranOut) > 0:
    maxRange = round(max(ranOut), 2)
    
data = np.array(map)

print(packer, len(out), coverage, maxRange)
plt.imshow(data)
plt.title("Coverage: " + str(coverage))
plt.show()
