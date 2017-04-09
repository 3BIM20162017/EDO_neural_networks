import numpy as np
import matplotlib.pyplot as plt

#http://stackoverflow.com/questions/27994660/eulers-method-in-python
#https://sites.math.washington.edu/~wcasper/math307_win16/review/euler_method/euler_method.pdf
def euler(dx_dt,x0,dt,t_max):
	ts = np.arange(0,t_max,dt)
	x = np.zeros((ts.size,x0.size))
	x[0] = x0
	for i,t in enumerate(ts[1:]):
		x[i+1,:] = x[i] + dx_dt(x[i],t) * dt
	return ts,x

def dx_dt_test(xt,t):
	return -xt

"""
x0 = np.asarray([2,3,10])
ts,x = euler(dx_dt_test,x0,0.1,10)
plt.plot(ts,x)
plt.show()
"""

def F(x):
	return np.tanh(x)

def dx_dt_full(xt,t,tau,g,W,F=F):
	return 1./tau * (-xt + F(g*np.dot(xt,W)))

from functools import partial

"""
tau = 10.
g = 1.
N = 10
k = -1.0
x0 = np.random.random(N) - 0.5
W = np.ones((N,N)) * k
dx_dt = partial(dx_dt_full,tau=tau,g=g,W=W)
ts,x = euler(dx_dt,x0,0.1,100)
plt.plot(ts,x)
plt.show()
"""

tau = 10.
g = 0.5
N = 20
x0 = np.random.random(N) - 0.5
W = np.random.randn(N,N)

plt.figure(1)
taulist = [0.5,1,4,8,12,20]
for i,taui in enumerate(taulist):
	plt.subplot(231+i)
	dx_dt = partial(dx_dt_full,tau=taui,g=g,W=W)
	ts,x = euler(dx_dt,x0,0.01,200)
	plt.plot(ts,x)
plt.show()

plt.figure(2)
glist = [0,2,4,6,8,10]
for i,gi in enumerate(glist):
	plt.subplot(231+i)
	dx_dt = partial(dx_dt_full,tau=10.,g=gi,W=W)
	ts,x = euler(dx_dt,x0,0.01,200)
	plt.plot(ts,x)
plt.show()

