import numpy as np
from matplotlib import pyplot as plt
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
            print(''.join(string))
            self.map[''.join(string)] = icomb

def load_ham(fname):
    h = np.array(h5py.File(f'{fname}.h5', 'r')['ham']['matrix'])
    token = hashlib.sha1(h.view(np.uint8)).hexdigest()
    try:
        with open(f'{fname}.cache.pkl', 'rb') as f: cached = pkl.load(f)
        if cached['token']!=token:
            print('cache out of date')
            raise Exception
        return cached['evals'], cached['evecs']
    except:
        print('eigenvalues/vectors are uncached - calculating...')
        evals, evecs = np.linalg.eigh(h)
        with open(f'{fname}.cache.pkl', 'wb') as f: 
            d = {'token': token, 'evals': evals, 'evecs': evecs}
            pkl.dump(d, f)
        return evals, evecs

evals, evecs = load_ham('ham_14')
basis = SpinBasis(14, 7)

print(evals[:10])
ineel = basis.map['10101010101010']
print(ineel)
assert ineel==1078
neel_coeffs = evecs[ineel, :]
print(neel_coeffs[:10])
assert abs(np.linalg.norm(neel_coeffs)-1.0)<1e-8


assert 0

w_max = 100
nsamp_w = 10000
w = np.linspace(0, w_max, nsamp_w)
t = np.fft.rfftfreq(len(w), d=w[1]-w[0])

'''
normalized to 2pi
'''
def norm_gaussian(x, sigma, mu):
    return np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)/sigma
    #return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def lorentzian(x, sigma, mu):
    return np.pi*sigma / (sigma**2 + (x-mu)**2)
    #return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

peak_fn = lorentzian

def cw(width, shift):
    cw = np.zeros_like(w)
    for i, e in enumerate(evals): 
        cw += peak_fn(w, width, e+shift-evals[0])*abs(neel_coeffs[i])**2
    #area = np.trapz(cw, w)
    #area_chk = np.dot(neel_coeffs, neel_coeffs)*2*np.pi
    #print(area, area_chk)
    return cw

def gt(width, shift):
    gt = np.fft.irfft(cw(width, shift), n=len(t))
    return gt/gt[0]
