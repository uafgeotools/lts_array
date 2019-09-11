#%% module imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import sys
from obspy import read, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from ltsva import ltsva
import flts_helper_array as fltsh
from plotting import lts_array_plot
from flts_helper_array import getrij

# A Bogoslof Explosion recorded at the AVO Adak Infrasound Array

#%% Read in and filter data





# Array Parameters
NET='AV'
STA = 'ADKI'
CHAN = 'BDF'
LOC = '01,02,03,04,05,06'

#note IRIS doesn't currently have data for June 2017!
STARTTIME = UTCDateTime('2019-06-10T13:10')
ENDTIME = STARTTIME + 20*60

st=Stream()

print('Reading in data from IRIS')
client = Client("IRIS")
st = client.get_waveforms(NET,STA,LOC,CHAN,STARTTIME,ENDTIME,attach_response=True)
st.merge(fill_value='latest')
st.trim(STARTTIME,ENDTIME,pad='true',fill_value=0)
st.sort()

print(st)
    
fs=st[0].stats.sampling_rate

print('Removing sensitivity...')
st.remove_sensitivity()


# st = read('Bogoslof_Data.mseed')
# calib = 4.7733e-05

# Filtering the data [Hz]
fmin = 0.5
fmax = 2.0
stf = st.copy()
stf.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=6, zerophase=True)
stf.taper


#%% get inventory and lat/lon info
inv = client.get_stations(network=NET,station=STA,channel=CHAN,location=LOC,
    starttime=STARTTIME,endtime=ENDTIME, level='channel')

latlist=[]
lonlist=[]
staname=[]
for network in inv:
    for station in network:
        for channel in station:
            latlist.append(channel.latitude)
            lonlist.append(channel.longitude)
            staname.append(channel.code)

rij=getrij(latlist,lonlist) #get element rijs


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
    fig10 = plt.figure(11)
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
    data[:, ii] = tr.data

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


