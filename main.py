import random
import numpy as np
import seaborn
import matplotlib.pyplot as plt

timestep = 0.01
time = 40
timevector = np.arange(0,time+timestep,timestep)
size = len(timevector)

runtimes = 50

mass = 10000  #weight in kg

g = -9.8


Fx = 0
Fy = 0
Fz = 0

Mx = 0
My = 0
Mz = 0

I = np.array([[100,0,0],[0,100,0],[0,0,100]])

u = 0
v = 0
w = 0

p = 0
q = 0
r = 0

theta = 0
phi = 0
psi = 0

x = 0
y = 0
z = 0

thetadot = 0
phidot = 0
psidot = 0

##set is the input array
xset = [0, 40, 100]
yset = [0, 0.6, 3.8]
zset = [0, -20, -96]

'''
Fxset = 0.5*0.2*0.001*3*(np.random.normal(0.1, 0.1, runtimes)**2)
Fyset = 0.5*0.2*0.001*3*(np.random.normal(0.1, 0.1, runtimes)**2)
Fzset = np.full(runtimes,-10000*g)+0.5*0.2*0.001*3*(np.random.normal(0, 0.2, runtimes)**2)

'''
Fxset = [0,0,0]
Fyset = [0,0,0]
Fzset = np.full(3,-9980*g)


Mxset = [0,0,0]
Myset = [0,0,0]
Mzset = [0,0,0]

uset = [1,1.5,2.5]
vset = [0,0,0]
wset = [0,-1,-2]

pset = [0,0,0]
qset = [0,0,0]
rset = [0,0,0]

thetaset = [0,0,0]
phiset = [0,0,0]
psiset = [np.pi*(1/180),np.pi*(3/180),np.pi*(4/180)]

thetadotset = [0,0,0]
phidotset = [0,0,0]
psidotset = [0,0,0]




def referforce():
    global Fx
    Fx = -0.5*0.2*0.001*3*(u**2) + Fxset[j]


def setforce(j):
    global Fx
    global Fy
    global Fz
    Fx = Fxset[j]
    Fy = Fyset[j]
    Fz = Fzset[j]


def setinputnum(j):
    global Fx
    global Fy
    global Fz
    global u
    global v
    global w
    global p
    global g
    global q
    global r
    global theta
    global phi
    global psi
    global thetadot
    global phidot
    global psidot
    global Mx
    global My
    global Mz
    global x
    global y
    global z

    Fx = Fxset[j]
    Fy = Fyset[j]
    Fz = Fzset[j]

    Mx = Mxset[j]
    My = Myset[j]
    Mz = Mzset[j]

    u = uset[j]
    v = vset[j]
    w = wset[j]

    p = pset[j]
    q = qset[j]
    r = rset[j]

    theta = thetaset[j]
    phi = phiset[j]
    psi = psiset[j]

    x = xset[j]
    y = yset[j]
    z = zset[j]

    thetadot = thetaset[j]
    phidot = phiset[j]
    psidot = psidotset[j]



def g_to_b(theta,phi,psi):
    output = np.array([[np.cos(theta)*np.cos(psi),np.cos(theta)*np.sin(psi),-1*np.sin(theta)],
                       [np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi),
                        np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi),
                        np.sin(phi)*np.cos(theta)],
                       [np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi),
                        np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi),
                        np.cos(phi)*np.cos(theta)]])
    return output


def e_to_b(theta,phi,psi):
    output = np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
                       [0,np.cos(phi),-1*np.sin(phi)],
                       [0,np.sin(phi)*(1/np.cos(theta)),np.cos(phi)*(1/np.cos(theta))]])
    return output



def updatecalculation():
    global u
    global v
    global w
    global p
    global q
    global r
    global Fx
    global Fy
    global Fz
    global mass
    global theta
    global phi
    global psi
    global g
    global Mx
    global My
    global Mz
    global I
    global thetadot
    global phidot
    global psidot
    global x
    global y
    global z

    a_vector = (np.cross(np.array([[u],[v],[w]]).T,np.array([[p],[q],[r]]).T).T + np.array([[Fx],[Fy],[Fz]]) * (1/mass)+
                g_to_b(theta,phi,psi) @ np.array([[0],[0],[g]]))


    angular_a_vector = np.linalg.inv(I) @ (-1 * np.cross(np.array([[p],[q],[r]]).T, (I @ np.array([[p],[q],[r]])).T).T +
                                         np.array([[Mx],[My],[Mz]]))

    u = u + a_vector[0, 0] * timestep
    v = v + a_vector[1, 0] * timestep
    w = w + a_vector[2, 0] * timestep

    p = p + angular_a_vector[0, 0] * timestep
    q = q + angular_a_vector[1, 0] * timestep
    r = r + angular_a_vector[2, 0] * timestep

    angle_dot_vector = e_to_b(theta,phi,psi) @ np.array([[p],[q],[r]])

    vlocity_vector = g_to_b(theta,phi,psi).T @ np.array([[u],[v],[w]])

    x = x + vlocity_vector[0, 0] * timestep
    y = y + vlocity_vector[1, 0] * timestep
    z = z + vlocity_vector[2, 0] * timestep


    phi = phi + angle_dot_vector[0,0] * timestep
    theta = theta + angle_dot_vector[1, 0] * timestep
    psi = psi + angle_dot_vector[2, 0] * timestep


x_record = np.array([x])
y_record = np.array([y])
z_record = np.array([z])

def record_data():
    global x_record
    global y_record
    global z_record
    global x
    global y
    global z

    x_record = np.append(x_record, x)
    y_record = np.append(y_record, y)
    z_record = np.append(z_record, z)


for j in range(len(xset)):
    setinputnum(j)
    for i in range(size):
        updatecalculation()
        record_data()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D View')

plt.xlabel('X position')
plt.ylabel('Y position')
plt.clabel('Z position')


fig2 = plt.figure()
xy = fig2.add_subplot(131)
xy.set_title('xy View')
plt.xlabel('X position')
plt.ylabel('Y position')




xz = fig2.add_subplot(132)
xz.set_title('xz View')
plt.xlabel('X position')
plt.ylabel('Z position')



yz = fig2.add_subplot(133)
yz.set_title('yz View')
plt.xlabel('Y position')
plt.ylabel('Z position')


for i in range(len(x_record)):
    color = plt.cm.viridis(i / len(x_record))
    ax.plot(x_record[i:i+2], y_record[i:i+2], z_record[i:i+2], color = color)
    xy.plot(x_record[i:i + 2], y_record[i:i + 2], color=color)
    xz.plot(x_record[i:i + 2], z_record[i:i + 2], color=color)
    yz.plot( y_record[i:i + 2], z_record[i:i + 2], color=color)

plt.tight_layout()
'''



finalxdata = np.zeros(runtimes)
finalydata = np.zeros(runtimes)
finalzdata = np.zeros(runtimes)

for j in range(runtimes):
    setinputnum(0)
    setforce(j)
    for i in range(size):
        updatecalculation()
        record_data()
    finalxdata[j] = x_record[-1]
    finalydata[j] = y_record[-1]
    finalzdata[j] = z_record[-1]

    x_record = np.array([x]) ##freememory
    y_record = np.array([y])
    z_record = np.array([z])


    print(j)

seaborn.kdeplot(x=finalxdata, y=finalydata, cmap="viridis", fill=True, thresh=0, levels=100)
plt.xlabel('X position')
plt.ylabel('Y position')
plt.gca().set_facecolor(plt.cm.viridis(1))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(0, 0, 0, color='red', s=100)

plt.xlabel('X position')
plt.ylabel('Y position')
plt.clabel('Z position')

for i in range(runtimes):
    color = plt.cm.viridis(i / runtimes)
    ax.scatter(finalxdata[i], finalydata[i], finalzdata[i], color = color,s = 0.5)
'''





plt.show()