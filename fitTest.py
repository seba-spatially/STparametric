import numpy as np
import pandas as pd
from numpy import linalg
import statsmodels.api as sm
data = pd.read_csv('oneDeviceOneDay.tsv', sep='\t')
data = data[['deviceTime', 'lastSeen', 'point','accuracy']]
p = [coordParser(x) for x in data.point]
x = [float(i[0]) for i in p]
y = [float(i[1]) for i in p]
data['x'] = x
data['y'] = y
del data['point']
data.head()
data = data.sort_values(by='deviceTime', axis=0)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')
d= data.ix[6:10]
L = range(0,len(d))
ax.scatter(d.x, d.y, d.deviceTime, c='r')

px = np.polyfit(d.x, L, deg=1, w=1/d.accuracy, cov=True)
np.sqrt(np.diagonal(px[1]))

py = np.polyfit(d.y,L,deg=1,w=1/d.accuracy)
pz = np.polyfit(d.deviceTime,L,deg=1,w=1/d.accuracy)



Ax = np.array([d.x, np.ones(len(L))]).T
Ay = np.array([d.y, np.ones(len(L))]).T
Az = np.array([d.deviceTime, np.ones(len(L))]).T

Wxy = np.diag(1/d.accuracy)
Wz = np.diag(np.ones(len(L)))
import timeit
start_time = timeit.default_timer()
np.polyfit(d.x,L,deg=1,w=1/d.accuracy)
elapsed = timeit.default_timer() - start_time
print(elapsed)
np.matmul(np.matmul(np.matmul(linalg.inv(np.matmul(np.matmul(Ax.T,Wxy),Ax)),Ax.T),Wxy),L)
elapsed = timeit.default_timer() - start_time
print(elapsed)


len(d)
X = []
Y = []
Z = list(np.linspace(1,50,5))

for i in range(0,100,20):
    d = i*np.pi/180.0
    Y.append(np.sin(d))
    X.append(np.cos(d))

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

L = np.array([list(range(len(Z)))]).T
#independent fits
# ax**2 + bx +c
Ax = np.array([X**2, X, np.ones(len(X))]).T
Ay = np.array([Y**2, Y, np.ones(len(Y))]).T
Az = np.array([Z**2, Z, np.ones(len(Z))]).T

#Z as a function of the index
cz = np.matmul(linalg.pinv(Az),L)
cx = np.matmul(linalg.pinv(Ax),L)
cy = np.matmul(linalg.pinv(Ay),L)

#predict L for Z = [1/3,2/3]
z1 = np.min(Z)+(np.max(Z)-np.min(Z))*1/3
z2 = np.min(Z)+(np.max(Z)-np.min(Z))*2/3

l1 = cz[0]*z1**2 + cz[1]*z1 + cz[2]
l2 = cz[0]*z2**2 + cz[1]*z2 + cz[2]
[l1,l2]


tt = np.linspace(np.min(L),np.max(L),100)
fx = []
fy = []
for t in tt:
    fx.append(cx[0] * t ** 2 + cx[1] * t + cx[2])
    fy.append(cy[0] * t ** 2 + cy[1] * t + cy[2])

x1 = tt[np.argmin(abs(fx-l1))]
x2 = tt[np.argmin(abs(fx-l2))]
[x1,x2]

y1 = tt[np.argmin(abs(fy-l1))]
y2 = tt[np.argmin(abs(fy-l2))]
[y1,y2]

u = 1/3
v = 2/3

a = 3 * (1 - u) * (1 - u) * u
b = 3 * (1 - u) * u * u
c = 3 * (1 - v) * (1 - v) * v
d = 3 * (1 - v) * v * v

det = a * d - b * c

p0x = X[0]
p0y = Y[0]
p0z = Z[0]
p3x = X[-1]
p3y = Y[-1]
p3z = Z[-1]


q1x = x1 - ((1 - u) * (1 - u) * (1 - u) * p0x + u * u * u * p3x)
q1y = y1 - ((1 - u) * (1 - u) * (1 - u) * p0y + u * u * u * p3y)
q1z = z1 - ((1 - u) * (1 - u) * (1 - u) * p0z + u * u * u * p3z)
[q1x,q1y,q1z]

q2x = x2 - ((1 - v) * (1 - v) * (1 - v) * p0x + v * v * v * p3x)
q2y = y2 - ((1 - v) * (1 - v) * (1 - v) * p0y + v * v * v * p3y)
q2z = z2 - ((1 - v) * (1 - v) * (1 - v) * p0z + v * v * v * p3z)
[q2x,q2y,q2z]

p1x = d * q1x - b * q2x
p1y = d * q1y - b * q2y
p1z = d * q1z - b * q2z
p1x /= det
p1y /= det
p1z /= det
[p1x,p1y,p1z]

p2x = (-c) * q1x + a * q2x
p2y = (-c) * q1y + a * q2y
p2z = (-c) * q1z + a * q2z
p2x /= det
p2y /= det
p2z /= det
[p2x,p2y,p2z]

tt = np.linspace(0,1,100)

bez = []
for t in tt:
    x = (1-t)**3 * p0x + 3*(1-t)**2*t*p1x + 3*(1-t)*t**2*p2x + t**3*p3x
    y = (1-t)**3 * p0y + 3*(1-t)**2*t*p1y + 3*(1-t)*t**2*p2y + t**3*p3y
    z = (1-t)**3 * p0z + 3*(1-t)**2*t*p1z + 3*(1-t)*t**2*p2z + t**3*p3z
    bez.append([x,y,z])


bez = pd.DataFrame(bez,columns=['x','y','z'])
bez.head()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='r')
ax.scatter([x1,x2], [y1,y2], [z1,z2], c='b')
ax.scatter([p0x,x1,x2,p3x], [p0y,y1,y2,p3y], [p0z,z1,z2,p3z], c='g')
ax.scatter(bez.x,bez.y,bez.z,c='k')
