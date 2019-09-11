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


#%% Read in and filter data

# Array Parameters
NET='AV'
STA = 'ADKI'
CHAN = '*DF'
LOC = '*'

#note IRIS doesn't currently have data for June 2017 but will soon
#STARTTIME = UTCDateTime('2017-06-10T13:10')
STARTTIME = UTCDateTime('2019-8-13T19:50')
ENDTIME = STARTTIME + 10*60

# Filter limits
FMIN = .5
FMAX = 5

# Processing parameters
WINLEN = 30
WINOVER = 0.50


#%%
st=Stream()

print('Reading in data from IRIS')
client = Client("IRIS")
st = client.get_waveforms(NET,STA,LOC,CHAN,STARTTIME,ENDTIME,attach_response=True)
st.merge(fill_value='latest')
st.trim(STARTTIME,ENDTIME,pad='true',fill_value=0)
st.sort()
print(st)

print('Removing sensitivity...')
st.remove_sensitivity()

stf = st.copy()
stf.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
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

# Plot array coords as a check
plotarray = 1
if plotarray:
    fig10 = plt.figure(11)
    plt.clf()
    plt.plot(rij[0, :], rij[1, :], 'ro')
    plt.axis('equal')
    plt.ylabel('km')
    plt.xlabel('km')
    plt.title(stf[0].stats.station)
    for i in range(len(stf)):
        plt.text(rij[0, i], rij[1, i], stf[0].stats.location)


# Parameters from the stream file
tvec=dates.date2num(st[0].stats.starttime.datetime)+st[0].times()/86400
nchans = len(stf)
npts = stf[0].stats.npts
fs=stf[0].stats.sampling_rate

data = np.empty((npts, nchans))
for ii, tr in enumerate(stf):
    data[:, ii] = tr.data

#%% Run LTS array processing

winlensamp = int(WINLEN*fs)
sampinc = int((1-WINOVER)*winlensamp)

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


