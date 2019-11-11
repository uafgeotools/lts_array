#%% module imports
import matplotlib.pyplot as plt

from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

import flts_helper_array as fltsh
from ltsva import ltsva
from plotting import lts_array_plot

# Read in and filter data
# Array Parameters
NET = 'AV'
STA = 'ADKI'
CHAN = '*DF'
LOC = '*'

# Pick an event
STARTTIME = UTCDateTime('2019-8-13T19:50')
ENDTIME = STARTTIME + 10*60

# Filter limits
FMIN = 0.5
FMAX = 5.0

# Processing parameters
WINLEN = 30
WINOVER = 0.50
# LTS alpha parameter - subset size
ALPHA = 0.5

#%%
st = Stream()

print('Reading in data from IRIS')
client = Client("IRIS")
st = client.get_waveforms(NET, STA, LOC, CHAN,
                          STARTTIME, ENDTIME, attach_response=True)
st.merge(fill_value='latest')
st.trim(STARTTIME, ENDTIME, pad='true', fill_value=0)
st.sort()
print(st)

print('Removing sensitivity...')
st.remove_sensitivity()

stf = st.copy()
stf.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=2, zerophase=True)
stf.taper


#%% Get inventory and lat/lon info
inv = client.get_stations(network=NET, station=STA, channel=CHAN,
                          location=LOC, starttime=STARTTIME,
                          endtime=ENDTIME, level='channel')

latlist = []
lonlist = []
staname = []
for network in inv:
    for station in network:
        for channel in station:
            latlist.append(channel.latitude)
            lonlist.append(channel.longitude)
            staname.append(channel.code)

# Get element rijs
rij = fltsh.getrij(latlist, lonlist)

# Plot array coords as a check
plotarray = 1
if plotarray:
    fig0 = plt.figure(10)
    plt.clf()
    plt.plot(rij[0, :], rij[1, :], 'ro')
    plt.axis('equal')
    plt.ylabel('km')
    plt.xlabel('km')
    plt.title(stf[0].stats.station)
    for ii in range(len(stf)):
        plt.text(rij[0, ii], rij[1, ii], stf[ii].stats.location)


#%% Run LTS array processing
stdict, t, mdccm, LTSvel, LTSbaz, sigma_tau = ltsva(stf, rij, WINLEN, WINOVER, ALPHA)

#%% plotting
fig1, axs1 = lts_array_plot(stf, stdict, t, mdccm, LTSvel, LTSbaz)
