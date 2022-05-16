import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import h5py
from itertools import combinations as combs
import pickle as pkl
import hashlib

class SpinBasis:
    def __init__(self, nsite, nalpha):
        self.map = {}
        for icomb, comb in enumerate(combs(range(14), 7)):
            string = ['1']*nsite
            for i in comb: string[nsite-1-i] = '0'
            self.map[''.join(string)] = icomb

def load_ham(fname):
    h = np.array(h5py.File(f'{fname}.h5', 'r')['ham']['matrix'])
    token = hashlib.sha1(h.view(np.uint8)).hexdigest()
    try:
        with open(f'{fname}.cache.pkl', 'rb') as f: cached = pkl.load(f)
        if cached['token']!=token:
            print('cache out of date')
            raise Exception
        return h, cached['evals'], cached['evecs']
    except:
        print('eigenvalues/vectors are uncached - calculating...')
        evals, evecs = np.linalg.eigh(h)
        with open(f'{fname}.cache.pkl', 'wb') as f: 
            d = {'token': token, 'evals': evals, 'evecs': evecs}
            pkl.dump(d, f)
        return h, evals, evecs

h, evals, evecs = load_ham('ham_14')
ndim = h.shape[0]
h -= np.eye(ndim)*evals[0]
evals -= evals[0]
basis = SpinBasis(14, 7)
ineel = basis.map['10101010101010']

dt = 0.01


class SliderParam:
    def __init__(self, name, init, lims):
        self.name = name
        self.init = init
        self.lims = lims
fig, ax = plt.subplots()

slider_params = [
    SliderParam('shift arg', np.pi/4, (0, np.pi/2)),
    SliderParam('shift mag', 1, (0, 10))
]

slider_axes = []
sliders = []
plt.subplots_adjust(bottom=0.25)

t = np.linspace(-10, 10, 1000)

for i, param in enumerate(slider_params):
    slider_axes.append(plt.axes([0.25, (i+1)*0.07, 0.65, 0.03]))
    sliders.append(Slider(ax=slider_axes[-1], label=param.name,
            valmin=param.lims[0], valmax=param.lims[1], valinit=param.init))

def f(t):
    theta = sliders[0].val
    r = sliders[1].val
    s = r*np.exp(-1j*theta)
    f = np.zeros(len(t), dtype=complex)
    for k, ek in enumerate(evals):
        #sk = complex(ek, 0)+s
        #f += evecs[ineel, k]*np.pi*sk**2*dt/(0.5 * dt**2 * (sk**4 + 2*(t - sk)**2))
        f += evecs[ineel, k]*np.pi*s/(s**2 + (t - ek)**2)
    return f

npole = 10
line_abs, = ax.plot(t, abs(f(t)), lw=2)

poles = []
for i in range(npole): 
    poles.append(ax.axvline(evals[i]-sliders[1].val, color='r'))

#line_real, = ax.plot(t, f(t).real, lw=2)
#line_imag, = ax.plot(t, f(t).imag, lw=2)

def update(val):
    f_t = abs(f(t))
    line_abs.set_ydata(f_t)
    #line_real.set_ydata(f_t.real)
    #line_imag.set_ydata(f_t.imag)
    ax.set_ylim(min(f_t)-1, max(f_t)+1)
    for ipole, pole in enumerate(poles): pole.set_xdata(evals[ipole]-sliders[1].val)
    fig.canvas.draw_idle()

for slider in sliders: slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    for slider in sliders: slider.reset()
button.on_clicked(reset)

plt.show()
