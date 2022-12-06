import numpy as np
import matplotlib.pyplot as plt
from obspy.geodetics.base import calc_vincenty_inverse


class DataBin:
    """ Data container for LTS processing

    Args:
        window_length (float): The window processing length [sec.]
        window_overlap (float): The decimal overalp [0.0, 1.0) for consecutive time windows.
        alpha (float): The decimal [0.50, 1.0] amount of data to keep in the subsets.
    """

    def __init__(self, window_length, window_overlap, alpha):
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.alpha = alpha

    def build_data_arrays(self, st, latlist, lonlist, remove_elements=None):
        """ Collect basic data from stream file. Project lat/lon into r_ij coordinates.

        Args:
            st (stream): An obspy stream object.
            latlist (list): A list of latitude points.
            lonlist (list): A list of longitude points.
            remove_elements (list): A list of elements to remove before processing. Python numbering is used, so "[0]" removes the first element.
        """
        # Check that all traces have the same length
        if len(set([len(tr) for tr in st])) != 1:
            raise ValueError('Traces in stream must have same length!')

        # Remove predetermined elements before processing
        if (remove_elements is not None) and (len(remove_elements) > 0):
            remove_elements = np.sort(remove_elements)
            for jj in range(0, len(remove_elements)):
                st.remove(st[remove_elements[jj]])
                latlist.remove(latlist[remove_elements[jj]])
                lonlist.remove(lonlist[remove_elements[jj]])
                remove_elements -= 1

        # Save the station name
        self.station_name = st[0].stats.station
        # Pull processing parameters from the stream file.
        self.nchans = len(st)
        self.npts = st[0].stats.npts
        # Save element names from location
        # If blank, pull from IMS-style station name
        self.element_names = []
        for tr in st:
            if tr.stats.location == '':
                # IMS element names
                self.element_names.append(tr.stats.station[-2:])
            else:
                self.element_names.append(tr.stats.location)
        # Assumes all traces have the same sample rate and length
        self.sampling_rate = st[0].stats.sampling_rate
        self.winlensamp = int(self.window_length * self.sampling_rate)
        # Sample increment (delta_t)
        self.sampinc = int(np.round(
            (1 - self.window_overlap) * self.winlensamp))
        # Time intervals to window data
        self.intervals = np.arange(0, self.npts - self.winlensamp, self.sampinc, dtype='int') # noqa
        self.nits = len(self.intervals)
        # Pull time vector from stream object
        self.tvec = st[0].times('matplotlib')
        # Store data traces in an array for processing.
        self.data = np.empty((self.npts, self.nchans))
        for ii, tr in enumerate(st):
            self.data[:, ii] = tr.data
        # Set the array coordinates
        self.rij = self.getrij(latlist, lonlist)
        # Make sure the least squares problem is well-posed
        # rij must have at least 3 elements
        if np.shape(self.rij)[1] < 3:
            raise RuntimeError('The array must have at least 3 elements for well-posed least squares estimation. Check rij array coordinates.')
        # Is least trimmed squares or ordinary least squares going to be used?
        if self.alpha == 1.0:
            print('ALPHA is 1.00. Performing an ordinary',
                  'least squares fit, NOT least trimmed squares.')

    def getrij(self, latlist, lonlist):
        r""" Calculate element locations (r_ij) from latitude and longitude.

        Return the projected geographic positions
        in X-Y (Cartesian) coordinates. Points are calculated
        with the Vincenty inverse and will have a zero-mean.

        Args:
            latlist (list): A list of latitude points.
            lonlist (list): A list of longitude points.

        Returns:
            (array):
            ``rij``: A numpy array with the first row corresponding to
            cartesian "X" - coordinates and the second row
            corresponding to cartesian "Y" - coordinates.

        """

        # Check that the lat-lon arrays are the same size.
        if (len(latlist) != self.nchans) or (len(lonlist) != self.nchans):
            raise ValueError('Mismatch between the number of stream channels and the latitude or longitude list length.') # noqa

        # Pre-allocate "x" and "y" arrays.
        xnew = np.zeros((self.nchans, ))
        ynew = np.zeros((self.nchans, ))

        for jj in range(1, self.nchans):
            # Obspy defaults to the WGS84 ellipsoid.
            delta, az, _ = calc_vincenty_inverse(
                latlist[0], lonlist[0], latlist[jj], lonlist[jj])
            # Convert azimuth to degrees from North
            az = (450 - az) % 360
            xnew[jj] = delta/1000 * np.cos(az*np.pi/180)
            ynew[jj] = delta/1000 * np.sin(az*np.pi/180)

        # Remove the mean.
        xnew -= np.mean(xnew)
        ynew -= np.mean(ynew)

        rij = np.array([xnew.tolist(), ynew.tolist()])

        return rij

    def plot_array_coordinates(self):
        """ Plot array element locations in Cartesian coordinates to the default device.
        """
        # Plot array coordinates
        fig = plt.figure(1)
        plt.clf()
        plt.plot(self.rij[0, :], self.rij[1, :], 'ro')
        plt.axis('equal')
        plt.ylabel('km')
        plt.xlabel('km')
        plt.title(self.station_name)
        for jj in range(0, self.nchans):
            plt.text(self.rij[0, jj], self.rij[1, jj], self.element_names[jj])
        fig.show()
