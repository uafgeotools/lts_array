import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import sys
from obspy import read
from ltsva import ltsva
import flts_helper_array as fltsh
from plotting import lts_array_plot


# A Bogoslof Explosion recorded at the AVO Adak Infrasound Array

#%% Read in and filter data
st = read('Bogoslof_Data.mseed')

# Array Parameters
arr = 'Adak Infrasound Array'
sta = 'ADKI',
chan = 'BDF'
loc = ['01', '02', '03', '04', '05', '06']
calib = 4.7733e-05

# Array Coordinates Projected into XY
rij = np.array([[0.0892929, 0.10716529, 0.03494914,
                 -0.043063, -0.0662987, -0.1220462],
                [-0.0608855, 0.0874639, -0.020600412169657,
                 0.00124259, 0.09052575, -0.09774634]])
rij[0, :] = rij[0, :] - rij[0, 0]
rij[1, :] = rij[1, :] - rij[1, 0]

# Filtering the data [Hz]
fmin = 0.5
fmax = 2.0
stf = st.copy()
stf.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=6, zerophase=True)
stf.taper

# Parameters from the stream file
nchans = len(stf)
# Construct the time vector
tvec = dates.date2num(stf[0].stats.starttime.datetime) + np.arange(0, stf[0].stats.npts / stf[0].stats.sampling_rate, stf[0].stats.delta)/float(86400) # noqa
npts = stf[0].stats.npts
dt = stf[0].stats.delta
fs = 1/dt

# Plot array coords as a check
plotarray = 1
if plotarray:
    fig10 = plt.figure(10)
    plt.clf()
    plt.plot(rij[0, :], rij[1, :], 'ro')
    plt.axis('equal')
    plt.ylabel('km')
    plt.xlabel('km')
    plt.title(arr)
    for i in range(nchans):
        # Stn lables
        plt.text(rij[0, i], rij[1, i], loc[i])


# Apply calibration value and format in a data matrix
calval = [4.01571568627451e-06, 4.086743142144638e-06,
          4.180744897959184e-06, 4.025542997542998e-06]
data = np.empty((npts, nchans))
for ii, tr in enumerate(stf):
    data[:, ii] = tr.data*calib

#%% Run LTS array processing

# Window length (sec)
winlen = 30
# Overlap between windows
winover = 0.50
# Converting to samples
winlensamp = int(winlen*fs)
sampinc = int((1-winover)*winlensamp)
its = np.arange(0, npts, sampinc)
nits = len(its)-1

# Pre-allocating Data Arrays
vel = np.zeros(nits)
vel.fill(np.nan)
az = np.zeros(nits)
az.fill(np.nan)
mdccm = np.zeros(nits)
mdccm.fill(np.nan)
t = np.zeros(nits)
t.fill(np.nan)
LTSvel = np.zeros(nits)
LTSvel.fill(np.nan)
LTSbaz = np.zeros(nits)
LTSbaz.fill(np.nan)
# Station Dictionary for Dropped LTS Stations
stdict = {}

print('Running wlsqva for %d windows' % nits)
for jj in range(nits):

    # Get time from middle of window, except for the end
    ptr = int(its[jj]), int(its[jj] + winlensamp)
    try:
        t[jj] = tvec[ptr[0]+int(winlensamp/2)]
    except:
        t[jj] = np.nanmax(t, axis=0)

    LTSbaz[jj], LTSvel[jj], flagged, ccmax, idx, _ = ltsva(
                    data[ptr[0]:ptr[1], :], rij, fs, 0.50)

    mdccm[jj] = np.median(ccmax)
    stns = fltsh.arrayfromweights(flagged, idx)

    # Stash some metadata for plotting
    if len(stns) > 0:
        tval = str(t[jj])
        stdict[tval] = stns
    if jj == (nits-1):
        stdict['size'] = nchans

    tmp = int(jj/nits*100)
    sys.stdout.write("\r%d%% \n" % tmp)
    sys.stdout.flush()

print('Done\n')


#%% plotting

fig1,axs1=lts_array_plot(stf,stdict,t,mdccm,LTSvel,LTSbaz)


