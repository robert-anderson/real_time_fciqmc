import numpy as np
from matplotlib import pyplot as plt
import h5py
from itertools import combinations as combs
import pickle as pkl
import hashlib
from scipy.sparse import csr_matrix


def swapped(tup, i, j):
    fn = lambda k: i if k==j else (j if k==i else k)
    return tuple(tup[fn(k)] for k in range(len(tup)))

def make_spin_basis(nsite, nalpha):
    d = {}
    for icomb, comb in enumerate(combs(range(nsite), nalpha)):
        spins = tuple(1 if i in comb else 0 for i in range(nsite))
        d[spins] = icomb
    return d

def make_heis_ham(basis, coupling=1.0):
    nsite = len(tuple(basis)[0])
    irows = []
    icols = []
    values = []
    def add_from_inds(irow, icol, value): 
        irows.append(irow)
        icols.append(icol)
        values.append(value)
    def add_from_onvs(src, dst, value):
        add_from_inds(basis[src], basis[dst], value)
    def add_diagonal(onv):
        irow = basis[onv]
        e
        add_from_inds(irow, irow, energy)

    lfn = lambda i : i-1 if i else nsite-1
    rfn = lambda i : 0 if i==nsite-1 else i+1
    for onv, ionv in basis.items():
        add_diagonal(onv)
        for i in range(nsite):
            j = 0 if i==nsite-1 else i + 1
            if onv[j]!=onv[i]: add_from_onvs(onv, swapped(onv, i, j), coupling)
    return csr_matrix((values, (irows, icols)), shape=(len(basis),)*2)

basis = make_spin_basis(6, 3)
h = make_heis_ham(basis)
print(h.toarray())
assert 0


assert 0

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

def cmplx(r, theta): return r*np.exp(1j*theta)


s = cmplx(9, 0.0*np.pi/2)
w = np.linspace(-1, 20, 10000)
aw = np.zeros(len(w), dtype=complex)
for k, ek in enumerate(evals):
    sk = complex(ek, 0)+s
    aw += evecs[ineel, k]*np.pi*sk**2*dt/(0.5 * dt**2 * (sk**4 + 2*(w - sk)**2))

plt.plot(w, abs(aw))
plt.show()


assert 0
dt = 0.04
h2 = np.dot(h, h)

static_damp = 0.0
quad_damp = 0.0

u1 = np.eye(ndim)-1j*dt*h
u2 = u1 - 0.5*h2*dt**2
u2 -= quad_damp*h2*dt**2
u2 -= np.eye(ndim)*static_damp
niter = 20000

norm = np.zeros(niter)
gt = np.zeros(niter, dtype=complex)
gs_ovlp = np.zeros(niter, dtype=complex)
v = np.zeros_like(u2[:,0])
v[ineel] = 1
v_init = v.copy()
with open('out.dat', 'w') as f:
    f.write('# 1. cycle number  2. time  3. L2 norm  4. init ovlp real  5. init ovlp imag  6. exact GS ovlp abs\n')
    for iiter in range(niter):
        v[:] = np.dot(u2, v)
        norm[iiter] = np.linalg.norm(v)
        gt[iiter] = np.vdot(v_init, v)
        gs_ovlp[iiter] = np.vdot(evecs[:,0], v)/norm[iiter]
        f.write(f'{iiter} {iiter*dt} {norm[iiter]} {gt[iiter].real} {gt[iiter].imag} {abs(gs_ovlp[iiter])}\n')
    

assert 0
v1 = np.zeros_like(u1[:,0])
v2 = np.zeros_like(u1[:,0])
v1.real = evecs[:,0]
v2[:] = v1[:]
norm1 = np.zeros(niter)
norm2 = np.zeros(niter)
for iiter in range(niter):
    v1[:] = np.dot(u1, v1)
    norm1[iiter] = np.linalg.norm(v1)
    v2[:] = np.dot(u2, v2)
    norm2[iiter] = np.linalg.norm(v2)
    if norm1[iiter]>1e10: break
    #print(norm1[iiter], norm2[iiter])
#plt.plot(np.arange(iiter)*dt, np.log(norm1[:iiter]), label='U1')
#plt.plot(np.arange(iiter)*dt, np.log(norm2[:iiter]), label='U2')
plt.plot(np.arange(iiter)*dt, norm1[:iiter], label='U1')
plt.plot(np.arange(iiter)*dt, norm2[:iiter], label='U2')
plt.yscale('log')
plt.legend()
plt.show()

assert 0
plt.plot(np.arange(niter)*dt, np.log(norm), label='U1')
plt.plot(np.arange(niter)*dt, np.log(norm), label='U2')


assert 0


shift = 10
evals-=evals[0]
evals+=shift


print(evals[:10])
print(ineel)
assert ineel==1078
neel_coeffs = evecs[ineel, :]
print(neel_coeffs[:10])
assert abs(np.linalg.norm(neel_coeffs)-1.0)<1e-8

w = np.linspace(-abs(max(evals)), abs(max(evals)), 100000)
aw = np.zeros_like(w)
dt = 0.001
for k, ek in enumerate(evals):
    if k<20: plt.axvline(ek, color='r')
    denom = 0.5*ek**4*dt**2 + 2*(w - ek)**2
    aw+=neel_coeffs[k]*(np.pi*ek**2*dt)/denom

plt.plot(w, abs(aw))
plt.xlim((shift-1, 2*shift))
plt.show()

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
