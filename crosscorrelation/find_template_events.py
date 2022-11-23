from glob import glob
import numpy as np
from obspy import Catalog
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, plot_trigger
from obspy.geodetics import gps2dist_azimuth
from seismonpy.io.nordic import read_nordic
from seismonpy.norsardb import Client
from multiprocessing import Pool
import scipy.signal
import scipy.cluster
from matplotlib import pyplot as plt




"""
Superseeded by crosscorrelation_detector.py
"""



ARCES_LAT, ARCES_LON = 69.53, 25.51
KIRUNA_LAT, KIRUNA_LON = 67.836, 20.173



def read_and_filter_bulletin(filename, sitename, min_mag):

    filtered_cat = Catalog()

    try:
        cat = read_nordic(filename, include_arrivals=False)
    except Exception as exc:
        print('In file {}: {}'.format(filename, exc))

    for ev in cat.events:
        if (ev.event_type == 'explosion' and
            #ev.event_type_certainty == 'known' and
            #ev.origins[0].evaluation_mode == 'manual' and
            ev.magnitudes[0].mag >= min_mag and
            any([sitename in c.text for c in ev.comments])
        ):
            filtered_cat.append(ev)
    return filtered_cat


def find_optimal_filters():
    """
    Try different filters on high-mag events to find the frequencies
    that give the highest SNR
    """
    pass


def load_waveform(dbclient, station, channel, starttime, length):

    edge_length = 2.0
    startt = starttime - edge_length
    endt = starttime + length + edge_length
    stream = dbclient.get_waveforms(station, channel=channel, starttime=startt, endtime=endt)

    stream.detrend('demean')
    stream.taper(max_percentage=None, max_length=edge_length, type='cosine', halfcosine=True)
    stream.filter('bandpass', freqmin=2.0, freqmax=8.0)

    stream.plot()

    return stream.traces[0].data


def load_beam(dbclient, stations, channel, baz, slowness, starttime, length, inventory, plot=False):

    edge_length = 2.0
    startt = starttime - edge_length
    endt = starttime + length + edge_length
    stream = dbclient.get_waveforms(stations, channel=channel, starttime=startt, endtime=endt)

    stream.detrend('demean')
    stream.taper(max_percentage=None, max_length=edge_length, type='cosine', halfcosine=True)
    stream.filter('bandpass', freqmin=3.0, freqmax=8.0)
    stream.rotate('NE->RT', back_azimuth=baz, inventory=inventory)

    p_time_delays = inventory.beam_time_delays(baz, slowness)
    beam = stream.create_beam(p_time_delays)
    beam.stats.channel = f'Beam, {channel}'
    
    if plot:
        beam.plot()

    return beam.data


def max_correlation(arr1, arr2):
    corr = scipy.signal.correlate(arr1, arr2, mode='full')
    return np.max(corr)



def cluster_on_reference_station():
    """
    Cluster events on reference station, as a first step in selecting template
    events 
    """

    SITENAME = 'KIRUNA'
    MIN_MAG = 2.0
    TIME_TO_ARCES = 40
    _, KIR_ARC_BAZ, _ = gps2dist_azimuth(ARCES_LAT, ARCES_LON, KIRUNA_LAT, KIRUNA_LON)
    print('KIR_ARC_BAZ', KIR_ARC_BAZ)

    
    catalog_path = '/staff/steffen/projects/external-catalogs/helsinki-catalog/nordic/'
    years = [2015, 2016, 2017, 2018]
    #years = [2015]#, 2016, 2017, 2018]

    data = []
    labels = []

    client = Client()
    inventory = client.get_array_inventory('ARCES', UTCDateTime(2015, 1, 1))

    for year in years:
        ypath = catalog_path + str(year)

        for nfile in glob(ypath + '/*.nordic'):
            #print('  Reading', nfile)
            cat = read_and_filter_bulletin(nfile, SITENAME, MIN_MAG)

            for event in cat:

                #print(event)
                #load_waveform(client, 'ARA0', 'BHZ', event.origins[0].time + TIME_TO_ARCES, 60)
                #continue

                label = str(event.origins[0].time) + ' ' + str(event.magnitudes[0].mag) + str(event.magnitudes[0].magnitude_type)
                labels.append(label)
                
                # Single trace
                #trace = load_waveform(client, 'ARA0', 'BHZ', event.origins[0].time + TIME_TO_ARCES, 60)
                #trace /= np.linalg.norm(trace)  # Required for correlation matrix
                #data.append(trace)

                # Beam
                beam = load_beam(client, 'AR*', 'BHZ', KIR_ARC_BAZ, 6.15, event.origins[0].time + TIME_TO_ARCES, 60, inventory)
                beam /= np.linalg.norm(beam)
                data.append(beam)



    # Compute correlation matrix
    correlation_threshold = 0.5
    n_obs = len(data)
    corr_matrix = np.zeros(shape=(n_obs, n_obs), dtype=np.float32)

    with Pool(processes=10) as pool:

        jobs = {}
        for i in range(n_obs):
            for j in range(n_obs):
                # Compute upper right triangle
                if j >= i:
                    jobs[(i, j)] = pool.apply_async(
                        max_correlation, (data[i], data[j])
                    )
        
        for i in range(n_obs):
            for j in range(n_obs):
                # Upper right triangle
                if j >= i:
                    maxcorr = jobs[(i, j)].get()
                    corr_matrix[i, j] = maxcorr

    for i in range(n_obs):
        for j in range(n_obs):
            # Lower left triangle
            if j < i:
                corr_matrix[i, j] = corr_matrix[j, i]

    # Zero the diagonal
    for i in range(n_obs):
        corr_matrix[i, i] = 0


    # Remove entries that poorly correlate with any other event
    max_per_column = np.max(corr_matrix, axis=0)
    above_thres = []
    above_thres_times = []
    for i in range(max_per_column.shape[0]):
        if max_per_column[i] > correlation_threshold:
            above_thres.append(i)
            above_thres_times.append(labels[i])
    
    print('Removing {} observations due to low correlation'.format(
        corr_matrix.shape[0] - len(above_thres)
    ))

    corr_matrix = corr_matrix[:, above_thres]
    corr_matrix = corr_matrix[above_thres, :]
    labels = above_thres_times
    
    # Compute distances
    for i in range(corr_matrix.shape[0]):
        corr_matrix[i, i] = 1   # reset diagonals to 1
    dist = 1.0 - corr_matrix
    #print(dist)
    dist = scipy.spatial.distance.squareform(dist)

    linkage = scipy.cluster.hierarchy.linkage(dist, method='single', optimal_ordering=True)
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, orientation='left', distance_sort='ascending', labels=labels)
    plt.show()


if __name__ == '__main__':

    # Plot templates events
    TIME_TO_ARCES = 40
    _, KIR_ARC_BAZ, _ = gps2dist_azimuth(ARCES_LAT, ARCES_LON, KIRUNA_LAT, KIRUNA_LON)
    client = Client()
    inventory = client.get_array_inventory('ARCES', UTCDateTime(2015, 1, 1))
    for tt in [
        UTCDateTime('2015-10-13T23:28:50.4'),
        UTCDateTime('2015-08-28T23:36:48.1')
    ]:
        beam = load_beam(client, 'AR*', 'BHZ', KIR_ARC_BAZ, 6.15, tt + TIME_TO_ARCES, 60, inventory, plot=True)



    #cluster_on_reference_station()
