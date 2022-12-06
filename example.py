# %% module imports
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client

import lts_array

# User Inputs
# Filter limits [Hz]
FREQ_MIN = 0.5
FREQ_MAX = 5.0

# Window length [sec]
WINDOW_LENGTH = 30

# Window overlap decimal [0.0, 1.0)
WINDOW_OVERLAP = 0.50

# LTS alpha parameter - subset size
ALPHA = 0.5

# Plot array coordinates
PLOT_ARRAY_COORDINATES = False

#####################################################
# End User inputs
#####################################################
# A short signal recorded at the Alaska Volcano Observatory
# Adak (ADKI) Infrasound Array
NET = 'AV'
STA = 'ADKI'
CHAN = '*'
LOC = '*'
START = UTCDateTime('2019-8-13T19:50')
END = START + 10*60

# Download data from IRIS
print('Reading in data from IRIS')
client = Client("IRIS")
st = client.get_waveforms(NET, STA, LOC, CHAN,
                          START, END, attach_response=True)
st.merge(fill_value='latest')
st.trim(START, END, pad='true', fill_value=0)
st.sort()
print(st)

print('Removing sensitivity...')
st.remove_sensitivity()

# Filter the data
st.filter("bandpass", freqmin=FREQ_MIN, freqmax=FREQ_MAX, corners=2, zerophase=True)
st.taper(max_percentage=0.05)

#%% Get inventory and lat/lon info
inv = client.get_stations(network=NET, station=STA, channel=CHAN,
                          location=LOC, starttime=START,
                          endtime=END, level='channel')

lat_list = []
lon_list = []
staname = []
for network in inv:
    for station in network:
        for channel in station:
            lat_list.append(channel.latitude)
            lon_list.append(channel.longitude)
            staname.append(channel.code)


# Flip a channel for testing
st[3].data *= -1

# Run processing
lts_vel, lts_baz, t, mdccm, stdict, sigma_tau, conf_int_vel, conf_int_baz = lts_array.ltsva(st, lat_list, lon_list, WINDOW_LENGTH, WINDOW_OVERLAP, ALPHA, PLOT_ARRAY_COORDINATES)

# Plot the results
fig, axs = lts_array.tools.lts_array_plot(st, lts_vel, lts_baz, t, mdccm, stdict)
# Plot uncertainty estimates
axs[1].plot(t, lts_vel + conf_int_vel, c='gray', linestyle=':')
axs[1].plot(t, lts_vel - conf_int_vel, c='gray', linestyle=':')
axs[2].plot(t, lts_baz + conf_int_baz, c='gray', linestyle=':')
axs[2].plot(t, lts_baz - conf_int_baz, c='gray', linestyle=':')

"""
Note that our flipped element is dropped in both data
windows that include the signal.
"""
