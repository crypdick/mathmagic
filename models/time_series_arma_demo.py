"""
Created on Thu Jan  8 13:26:34 2015

@author: rkp

This code generates a random signal x, filters the signal using signal filter
hsig, as well as feedback filter hfdbk, thus generating signal y. It then attempts to
reconstruct hsig and hfdbk given x and y.

It uses the statsmodels glm toolkit.
"""

import numpy as np
import matplotlib.pyplot as plt
from time_series import Arma


# SIGNAL LENGTH
T = 10000
# SIGNAL FILTER LENGTH
Thsig = 160
# FEEDBACK FILTER LENGTH
Thfdbk = 160
# NOISE LEVEL
noise = .1


# create time vector for x, hsig, and hfdbk
t = np.arange(T)
thsig = np.arange(Thsig)
thfdbk = np.arange(Thfdbk)

# create random signal x
x = np.random.normal(0, 1, T)

# create hsig & hfdbk
hsig = np.exp(-thsig/40.)
hsig /= np.linalg.norm(hsig)

hfdbk = -np.sin(thfdbk/24.)*np.exp(-thfdbk/16.)
hfdbk /= np.linalg.norm(hfdbk)

# create generative arma model
modelg = Arma(sig_filter_lens=[Thsig], fdbk_filter_len=Thfdbk, noise=noise)

# set filters
modelg.sig_filters = [hsig]
modelg.fdbk_filter = hfdbk
# set constant
modelg.constant = .15

# generate y from x
print 'Generating y from x...'
y = modelg.gen([x])

# create arma model to be fitted
modelf = Arma(sig_filter_lens=[Thsig], fdbk_filter_len=Thfdbk, noise=noise)

# fit filters
print 'Fitting filters ...'
modelf.fit([x], y)

hsig_fit = modelf.sig_filters[0]
hfdbk_fit = modelf.fdbk_filter

# plot x & y
fig = plt.figure(facecolor='white', tight_layout=True)
axx = fig.add_subplot(2, 1, 1)
axy = axx.twinx()

axx.plot(t, x, c='r')

axx.set_xlabel('t')
axx.set_ylabel('x', color='r')

axy.plot(t, y, c='b')
axy.set_ylabel('y', color='b')

# plot true filters
ax_hsig = fig.add_subplot(2, 2, 3)
ax_hsig.plot(thsig, hsig, c='g')

ax_hsig.set_xlabel('t')
ax_hsig.set_ylabel('hsig (true)')

ax_hfdbk = fig.add_subplot(2, 2, 4)
ax_hfdbk.plot(thfdbk, hfdbk, c='g')

ax_hfdbk.set_xlabel('t')
ax_hfdbk.set_ylabel('hfdbk (true)')

# plot fitted filters
ax_hsig_fit = ax_hsig.twinx()
ax_hsig_fit.plot(thsig, hsig_fit, c='c')

ax_hsig_fit.set_ylabel('hsig (fit)')

ax_hfdbk_fit = ax_hfdbk.twinx()
ax_hfdbk_fit.plot(thfdbk, hfdbk_fit, c='c')

ax_hfdbk_fit.set_ylabel('hfdbk (fit)')

# print constant
print 'constant = %.3f' % modelf.constant