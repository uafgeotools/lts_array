import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from copy import deepcopy
from collections import Counter
import os


def lts_array_plot(st, lts_vel, lts_baz, array, t, mdccm, out_dir, stdict=None):
    '''
    Return a Least-trimmed squares array processing plot, including
        flagged element pairs.
    Plots first channel waveform, trace-velocity, back-azimuth,
        and LTS-flagged element pairs.

    Args:
        st (stream): Obspy stream. Assumes response has been removed.
        stdict (dict): Dictionary of flagged element pairs
            from the `fast_lts_array` function.
        t (array): Array of time values for each parameter estimate.
        mdccm (array): Array of median cross-correlation maximas.
        array (str): name of array
        lts_vel (array): Array of least-trimmed squares
            trace velocity estimates.
        lts_baz (array): Array of least-trimmed squares
            back-azimuths estimates.
        out_dir (string): path for png

    Returns:
        (tuple):
            ``fig1``: Output figure handle.
            ``axs1``: Output figure axes.

    Example:
        fig, axs = lts_array_plot(st, lts_vel, lts_baz, t, mdccm, stdict)
    '''

    # Specify the colormap.
    cm = 'RdYlBu_r'
    # Colorbar/y-axis limits for MdCCM.
    cax = (0.2, 1)
    # Specify the time vector for plotting the trace.
    tvec = st[0].times('matplotlib')

    # Check station dictionary input. It must be a non-empy dictionary.
    if not isinstance(stdict, dict) or not stdict:
        stdict = None

    # Determine the number and order of subplots.
    num_subplots = 3
    vplot = 1
    bplot = 2
    splot = bplot
    if stdict is not None:
        num_subplots += 1
        splot = bplot + 1

    # Start plotting.
    fig, axarr = plt.subplots(num_subplots, 1, sharex='col')
    fig.set_size_inches(10, 12)
    axs = axarr.ravel()
    axs[0].plot(tvec, st[0].data, 'k')
    axs[0].axis('tight')
    axs[0].set_ylabel('Pressure [Pa]')
    axs[0].text(0.15, 0.93, st[0].stats.station, horizontalalignment='center',
                verticalalignment='center', transform=axs[0].transAxes)
    cbaxes = fig.add_axes(
        [0.95, axs[splot].get_position().y0, 0.02,
         axs[vplot].get_position().y1 - axs[splot].get_position().y0])

    # Plot the trace velocity plot.
    sc = axs[vplot].scatter(t, lts_vel, c=mdccm,
                            edgecolors='k', lw=0.1, cmap=cm)
    axs[vplot].set_ylim(0.25, 0.45)
    axs[vplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs[vplot].set_ylabel('Trace Velocity\n [km/s]')

    #  Plot the back-azimuth estimates.
    sc = axs[bplot].scatter(t, lts_baz, c=mdccm,
                            edgecolors='k', lw=0.1, cmap=cm)
    axs[bplot].set_ylim(0, 360)
    axs[bplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    axs[bplot].set_ylabel('Back-azimuth\n [deg]')

    hc = plt.colorbar(sc, cax=cbaxes, ax=[axs[1], axs[2]])
    hc.set_label('MdCCM')

    # Plot dropped station pairs from LTS if given.
    if stdict is not None:
        ndict = deepcopy(stdict)
        n = ndict['size']
        ndict.pop('size', None)
        tstamps = list(ndict.keys())
        tstampsfloat = [float(ii) for ii in tstamps]

        # Set the second colormap for station pairs.
        cm2 = plt.get_cmap('binary', (n-1))
        initplot = np.empty(len(t))
        initplot.fill(1)

        axs[splot].scatter(np.array([t[0], t[-1]]),
                           np.array([0.01, 0.01]), c='w')
        axs[splot].axis('tight')
        axs[splot].set_ylabel('Element [#]')
        axs[splot].set_xlabel('UTC Time')
        axs[splot].set_xlim(t[0], t[-1])
        axs[splot].set_ylim(0.5, n+0.5)
        axs[splot].xaxis_date()
        axs[splot].tick_params(axis='x', labelbottom='on')

        # Loop through the stdict for each flag and plot
        for jj in range(len(tstamps)):
            z = Counter(list(ndict[tstamps[jj]]))
            keys, vals = z.keys(), z.values()
            keys, vals = np.array(list(keys)), np.array(list(vals))
            pts = np.tile(tstampsfloat[jj], len(keys))
            sc2 = axs[splot].scatter(pts, keys, c=vals, edgecolors='k',
                                     lw=0.1, cmap=cm2, vmin=0.5, vmax=n-0.5)

        # Add the horizontal colorbar for station pairs.
        p3 = axs[splot].get_position().get_points().flatten()
        p3 = axs[splot].get_position()
        cbaxes2 = fig.add_axes([p3.x0, p3.y0-0.08, p3.width, 0.02])
        hc2 = plt.colorbar(sc2, orientation="horizontal",
                           cax=cbaxes2, ax=axs[splot])
        hc2.set_label('Number of Flagged Element Pairs')

    axs[splot].xaxis_date()
    axs[splot].set_xlabel('UTC Time')
    
    # save
    fname_time = st[0].times(type="utcdatetime")
    filename = array+'_'+UTCDateTime.strftime(fname_time[0],'%Y%m%d-%H%M.png')
    full_path = os.path.join(out_dir,filename)
    plt.savefig(full_path,dpi=72,format='png')

    return fig, axs

def lts_array_plot_thumb(st, lts_vel, lts_baz, array, t, mdccm, out_dir, stdict=None):
    '''
    Return a Least-trimmed squares array processing plot, including
        flagged element pairs.
    Plots first channel waveform, trace-velocity, back-azimuth,
        and LTS-flagged element pairs.

    Args:
        st (stream): Obspy stream. Assumes response has been removed.
        stdict (dict): Dictionary of flagged element pairs
            from the `fast_lts_array` function.
        t (array): Array of time values for each parameter estimate.
        mdccm (array): Array of median cross-correlation maximas.
        array (str): name of array
        lts_vel (array): Array of least-trimmed squares
            trace velocity estimates.
        lts_baz (array): Array of least-trimmed squares
            back-azimuths estimates.
        out_dir (string): path for png

    Returns:
        (tuple):
            ``fig1``: Output figure handle.
            ``axs1``: Output figure axes.

    Example:
        fig, axs = lts_array_plot(st, lts_vel, lts_baz, t, mdccm, stdict)
    '''

    # Specify the colormap.
    cm = 'RdYlBu_r'
    # Colorbar/y-axis limits for MdCCM.
    cax = (0.2, 1)
    # Specify the time vector for plotting the trace.
    tvec = st[0].times('matplotlib')

    # Check station dictionary input. It must be a non-empy dictionary.
    if not isinstance(stdict, dict) or not stdict:
        stdict = None

    # Determine the number and order of subplots.
    num_subplots = 3
    vplot = 1
    bplot = 2
    splot = bplot
    if stdict is not None:
        num_subplots += 1
        splot = bplot + 1

    # Start plotting.
    fig, axarr = plt.subplots(num_subplots, 1, sharex='col')
    fig.set_size_inches(2.1,2.75)
    axs = axarr.ravel()
    
    # axis formatting
    axs[0].plot(tvec, st[0].data, 'k')
    axs[0].axis('tight')
    axs[0].xaxis_date()
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    
    #axs[0].axis('tight')
    #axs[0].set_ylabel('Pressure [Pa]')
    #axs[0].text(0.15, 0.93, st[0].stats.station, horizontalalignment='center',
    #            verticalalignment='center', transform=axs[0].transAxes)
    #cbaxes = fig.add_axes(
    #    [0.95, axs[splot].get_position().y0, 0.02,
    #     axs[vplot].get_position().y1 - axs[splot].get_position().y0])

    # Plot the trace velocity plot.
    sc = axs[vplot].scatter(t, lts_vel, c=mdccm,
                            edgecolors='k', lw=0.1, cmap=cm)
    axs[vplot].set_ylim(0.25, 0.45)
    axs[vplot].set_xlim(t[0], t[-1])
    axs[vplot].set_xticks([])
    axs[vplot].set_yticks([])
    
    sc.set_clim(cax)
    #axs[vplot].set_ylabel('Trace Velocity\n [km/s]')
    #axs[vplot].set_ylabel('Trace Velocity\n [km/s]')

    #  Plot the back-azimuth estimates.
    sc = axs[bplot].scatter(t, lts_baz, c=mdccm,
                            edgecolors='k', lw=0.1, cmap=cm)
    axs[bplot].set_ylim(0, 360)
    axs[bplot].set_xlim(t[0], t[-1])
    sc.set_clim(cax)
    #axs[bplot].set_ylabel('Back-azimuth\n [deg]')
    #axs[bplot].set_ylabel([])
    axs[bplot].set_xticks([])
    axs[bplot].set_yticks([])

    #hc = plt.colorbar(sc, cax=cbaxes, ax=[axs[1], axs[2]])
    #hc.set_label('MdCCM')

    # Plot dropped station pairs from LTS if given.
    if stdict is not None:
        ndict = deepcopy(stdict)
        n = ndict['size']
        ndict.pop('size', None)
        tstamps = list(ndict.keys())
        tstampsfloat = [float(ii) for ii in tstamps]

        # Set the second colormap for station pairs.
        cm2 = plt.get_cmap('binary', (n-1))
        initplot = np.empty(len(t))
        initplot.fill(1)

        axs[splot].scatter(np.array([t[0], t[-1]]),
                           np.array([0.01, 0.01]), c='w')
        axs[splot].axis('tight')
        #axs[splot].set_ylabel('Element [#]')
        #axs[splot].set_xlabel('UTC Time')
        axs[splot].set_xlim(t[0], t[-1])
        axs[splot].set_ylim(0.5, n+0.5)
        #axs[splot].xaxis_date()
        #axs[splot].tick_params(axis='x', labelbottom='on')
        
        #axs[splot].set_ylabel([])
        #axs[splot].set_xlabel([])
        axs[splot].set_xticks([])
        axs[splot].set_yticks([])

        # Loop through the stdict for each flag and plot
        for jj in range(len(tstamps)):
            z = Counter(list(ndict[tstamps[jj]]))
            keys, vals = z.keys(), z.values()
            keys, vals = np.array(list(keys)), np.array(list(vals))
            pts = np.tile(tstampsfloat[jj], len(keys))
            sc2 = axs[splot].scatter(pts, keys, c=vals, edgecolors='k',
                                     lw=0.1, cmap=cm2, vmin=0.5, vmax=n-0.5)

        # Add the horizontal colorbar for station pairs.
        p3 = axs[splot].get_position().get_points().flatten()
        p3 = axs[splot].get_position()
        #cbaxes2 = fig.add_axes([p3.x0, p3.y0-0.08, p3.width, 0.02])
        #hc2 = plt.colorbar(sc2, orientation="horizontal",
        #                   cax=cbaxes2, ax=axs[splot])
        #hc2.set_label('Number of Flagged Element Pairs')

    #axs[splot].xaxis_date()
    #axs[splot].set_xlabel('UTC Time')
    #axs[splot].set_xlabel([])
    
    # save
    fname_time = st[0].times(type="utcdatetime")
    filename = array+'_'+UTCDateTime.strftime(fname_time[0],'%Y%m%d-%H%M_thumb.png')
    full_path = os.path.join(out_dir,filename)
    plt.savefig(full_path,dpi=72,format='png')

    return fig, axs


