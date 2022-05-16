import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import exact_spectrum as exspec
from subprocess import Popen, PIPE

def read_from_neci(fname, gt_cols, max_t=None):
    data = np.loadtxt(fname)
    t = data[:,8]
    n = None
    if max_t is not None: n = np.argmax(t>=max_t)
    if not n: n = len(t)
    return (data[:n,8],)+tuple(data[:n,col] for col in gt_cols)

def read_avg_from_neci(fname, first_col, nrep, max_t=None):
    gt_cols = tuple(first_col+i*(i+1) for i in range(nrep))
    cols = read_from_neci(fname, gt_cols, max_t)
    avg = np.zeros_like(cols[1])
    for i in range(1, nrep+1): avg+=cols[i]
    return cols[0], avg/nrep

def rsync(machine, src, dst='.'):
    Popen(f'rsync {machine}:"{src}" "{dst}"', shell=True).wait()


# Define initial parameters
init_eshift = 0.3
#init_peakwidth = 0.03
init_peakwidth = 0.08

# Create the figure and the line that we will manipulate

fig, axes = plt.subplots(2, figsize=(8, 8))
plt.suptitle('14-site Heisenberg model Green\'s functions',fontsize=20)
axw = axes[0]
len_w = len(exspec.w)//4
linew, = axw.plot(exspec.w[:len_w], exspec.cw(init_peakwidth, init_eshift)[:len_w], lw=2)
axw.set_xlabel(r'$\omega$')
axw.set_ylabel(r'$\tilde{C}(\omega)$')
axt = axes[1]
len_t = len(exspec.t)//2
linet, = axt.plot(exspec.t[:len_t]*(4*np.pi), exspec.gt(init_peakwidth, init_eshift)[:len_t], lw=2, label='exact')
#neci_t, neci_gt = read_from_neci('/home/anderson/work/real_time/deterministic/from_pops/neel/fciqmc_stats', [21], exspec.t[len_t-1]*4*np.pi)
#rsync('alxmp02', '/scratch/anderson/real_time/nw_100000/fciqmc_stats', 'stoch')
rsync('localhost', '/home/anderson/tmp/real_time/static_damping/fciqmc_stats', 'stoch')
neci_t, neci_gt = read_from_neci('./stoch/fciqmc_stats', [21], exspec.t[len_t-1]*4*np.pi)
#neci_t, neci_gt = read_avg_from_neci('./stoch/fciqmc_stats', 21, 5, exspec.t[len_t-1]*4*np.pi)

neci_dt = neci_t[1]-neci_t[0]
neci_t = np.arange(2*len(neci_t))*neci_dt
neci_t = np.arange(len(neci_t))*neci_dt

neci_gt = np.pad(neci_gt, (0, len(neci_gt)))
neci_gt[len(neci_t)//2:] = neci_gt[len(neci_t)//2:0:-1]

linen, = axt.plot(neci_t[:len(neci_t)//2], neci_gt[:len(neci_t)//2], lw=2, label='FCIQMC (100k walkers)')
axt.legend()
axt.set_xlabel(r'$t$')
axt.set_ylabel(r'$G(t)$')


neci_w = np.fft.rfftfreq(len(neci_t), d=(neci_t[1]-neci_t[0])/(2*np.pi))
neci_cw = np.fft.rfft(neci_gt/10)
axw.plot(neci_w, neci_cw)


# adjust the main plot to make room for the sliders
plt.subplots_adjust(bottom=0.25)

# Make a horizontal slider to control the spectral shift
axeshift = plt.axes([0.25, 0.1, 0.65, 0.03])
eshift_slider = Slider(
    ax=axeshift,
    label='Spectral shift',
    valmin=0.0,
    valmax=1.0,
    valinit=init_eshift,
)

# Make a vertically oriented slider to control the amplitude
axwidth = plt.axes([0.25, 0.15, 0.65, 0.03])
width_slider = Slider(
    ax=axwidth,
    label="Peak width",
    valmin=0.001,
    valmax=0.2,
    valinit=init_peakwidth,
    orientation="horizontal"
)


# The function to be called anytime a slider's value changes
def update(val):
    linew.set_ydata(exspec.cw(width_slider.val, eshift_slider.val)[:len_w])
    linet.set_ydata(exspec.gt(width_slider.val, eshift_slider.val)[:len_t])
    #neci_t, neci_gt = read_from_neci('/home/anderson/work/real_time/deterministic/from_pops/neel/fciqmc_stats', exspec.t[len_t-1]*4*np.pi)
    #linen.set_ydata(neci_gt)
    #linen.set_xdata(neci_t)
    fig.canvas.draw_idle()


# register the update function with each slider
eshift_slider.on_changed(update)
width_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    eshift_slider.reset()
    width_slider.reset()
button.on_clicked(reset)

plt.show()
