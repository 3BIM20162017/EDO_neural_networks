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
dx_dt = partial(dx_dt_full,tau=tau,g=g,W=W)
ts,x = euler(dx_dt,x0,0.001,200)
plt.plot(ts,x)
plt.show()