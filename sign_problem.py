import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)

ndim = 25

def get_rp(mat):
    tmp = mat.real.copy()
    tmp[np.where(tmp<0)] = 0.0
    return tmp
def get_rm(mat):
    tmp = mat.real.copy()
    tmp[np.where(tmp>=0)] = 0.0
    return np.abs(tmp)
def get_ip(mat):
    tmp = mat.imag.copy()
    tmp[np.where(tmp<0)] = 0.0
    return tmp
def get_im(mat):
    tmp = mat.imag.copy()
    tmp[np.where(tmp>=0)] = 0.0
    return np.abs(tmp)

def make_lambda_1(mat):
    tmp = np.zeros_like(mat)
    tmp.real = get_rp(mat) - get_rm(mat)
    tmp.imag = get_ip(mat) - get_im(mat)
    return tmp

def make_lambda_2(mat):
    tmp = np.zeros_like(mat)
    tmp.real = get_rp(mat) - get_rm(mat)
    tmp.imag = -get_ip(mat) + get_im(mat)
    return tmp

def make_lambda_3(mat):
    tmp = np.zeros_like(mat)
    tmp.real = get_rp(mat) + get_rm(mat) - get_ip(mat) - get_im(mat)
    return tmp

def make_lambda_4(mat):
    tmp = np.zeros_like(mat)
    tmp.real = get_rp(mat) + get_rm(mat) + get_ip(mat) + get_im(mat)
    return tmp

def make_lambda(mat, i):
    if i==1: return make_lambda_1(mat)
    if i==2: return make_lambda_2(mat)
    if i==3: return make_lambda_3(mat)
    if i==4: return make_lambda_4(mat)
    assert 0, "invalid lambda index"

def make_spencer_lambda(mat, anti_sym):
    assert np.allclose(mat.imag, np.zeros_like(mat.imag))
    return get_rp(mat)+(-1.0 if anti_sym else 1.0)*get_rm(mat)

def integrate(u_mat, niter):
    v = np.ones_like(u_mat[:,0])
    v/= np.linalg.norm(v)
    dv = np.zeros_like(v)
    norm_stats = np.zeros(niter)
    re_norm_stats = np.zeros(niter)
    im_norm_stats = np.zeros(niter)
    re_annihil_stats = np.zeros(niter)
    im_annihil_stats = np.zeros(niter)
    for iiter in range(niter):
        norm_stats[iiter] = np.linalg.norm(v)
        re_norm_stats[iiter] = np.linalg.norm(v.real)
        im_norm_stats[iiter] = np.linalg.norm(v.imag)
        dv[:] = np.dot(u_mat, v)
        mask = np.where(np.sign(dv.real)!=np.sign(v.real))
        re_annihil_stats[iiter] = np.sum(np.minimum(np.abs(v.real[mask]), np.abs(dv.real[mask])))
        mask = np.where(np.sign(dv.imag)!=np.sign(v.imag))
        im_annihil_stats[iiter] = np.sum(np.minimum(np.abs(v.imag[mask]), np.abs(dv.imag[mask])))
        v+=dv
    return norm_stats, re_norm_stats, im_norm_stats, re_annihil_stats, im_annihil_stats

def make_u1(h_mat, dt):
    return -1j*dt*h_mat

def make_u2(h_mat, dt):
    return make_u1(h_mat, dt)-0.5*dt**2*np.dot(h_mat, h_mat)

h_stoq = -np.random.random((ndim, ndim))
#h_stoq[(np.arange(h_stoq.shape[0]),)*2] = np.random.random(ndim)-0.5
h_stoq += h_stoq.T
evals, evecs = np.linalg.eigh(h_stoq)
print(evals)
shift = evals[3]
h_stoq[(np.arange(h_stoq.shape[0]),)*2] -= shift
evals -= shift


dt = 0.001*np.exp(-1*1j*np.pi/2)
dt = complex(0, -0.001)
print(f'complex time step: {dt}')
u = make_u1(h_stoq, dt)
print(f'u total real mag: {np.linalg.norm(u.real.flatten())}')
print(f'u total imag mag: {np.linalg.norm(u.imag.flatten())}')
print('all evec_0 elements +ve: '+str(np.all(evecs[:,0]>=0)))
print('all evec_0 elements -ve: '+str(np.all(evecs[:,0]<=0)))

assert np.allclose(u, make_lambda_1(u))

niter = 100

#for i in (1, 2, 3):
#    norm = integrate(make_lambda(u, i), niter)[0]
#    norm = integrate(make_lambda(u, i), niter)[0]
#    plt.plot(norm, label=f'$\Lambda_{i}$')
for anti_sym in (True, False):
    series = integrate(make_spencer_lambda(u, anti_sym), niter)[0]
    plt.plot(series, label='anti sym: '+str(anti_sym))

plt.legend()
plt.show()
assert 0


#norm, re_annihil, im_annihil = integrate_u2(h_stoq, dt, 10000)
norm, re_norm, im_norm, re_annihil, im_annihil = integrate(u, 1000)
#plt.plot(re_norm)
#plt.plot(im_norm)
#plt.plot(norm)
plt.plot(re_annihil)
plt.plot(im_annihil)
plt.show()

assert 0
print(h_stoq)

assert 0
theta = np.pi/4
phase = np.exp(-1j*theta)
Trp = T_stoq 



