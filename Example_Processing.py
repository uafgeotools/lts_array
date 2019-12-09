#%% module imports
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Import the package.
import lts_array

#%% Read in and filter data
# Array Parameters
NET = 'AV'
STA = 'ADKI'
CHAN = '*DF'
LOC = '*'

# Start and end of time window containing (suspected) events
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
stf.taper(max_percentage==0.05)


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
rij = lts_array.getrij(latlist, lonlist)

# Plot array coordinates as a check
fig1 = plt.figure(1)
plt.clf()
plt.plot(rij[0, :], rij[1, :], 'ro')
plt.axis('equal')
plt.ylabel('km')
plt.xlabel('km')
plt.title(stf[0].stats.station)
for i, tr in enumerate(stf):
    plt.text(rij[0, i], rij[1, i], tr.stats.location)

#%% Run LTS array processing
stdict, t, mdccm, LTSvel, LTSbaz, sigma_tau = lts_array.ltsva(stf, rij, WINLEN, WINOVER, ALPHA)

#%% Plotting
fig2, axs2 = lts_array.lts_array_plot(stf, stdict, t, mdccm, LTSvel, LTSbaz)
