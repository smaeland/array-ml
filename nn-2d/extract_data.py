import os
from glob import glob
import pickle
from argparse import ArgumentParser
import numpy as np
from obspy import Catalog
from obspy.core import stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, plot_trigger
from obspy.geodetics import gps2dist_azimuth
from seismonpy.io.nordic import read_nordic
from seismonpy.norsardb import Client
from multiprocessing import Pool



def get_site_name(event):
    for comm in event.comments:
        if comm.text.startswith('MINING AREA'):
            site = comm.text.split()[-2]
            return site.strip()


def read_and_filter_bulletin(filename, sitenames, min_mag):

    filtered_cat = Catalog()

    try:
        cat = read_nordic(filename, include_arrivals=False)
    except Exception as exc:
        print('In file {}: {}'.format(filename, exc))

    for ev in cat.events:
        if (ev.event_type == 'explosion' and
            ev.event_type_certainty == 'known' and
            #ev.origins[0].evaluation_mode == 'manual' and
            ev.magnitudes[0].mag >= min_mag
        ):
            if get_site_name(ev) in sitenames:
                filtered_cat.append(ev)
            
            #for comm in ev.comments:
            #    for site in sitenames:
            #        if site in comm.text:
            #            filtered_cat.append(ev)
    
    return filtered_cat


def create_event_list(years, min_magnitude, sites):

    catalog_path = '/staff/steffen/projects/external-catalogs/helsinki-catalog/nordic/'
    catalog = Catalog()

    with Pool(processes=8) as pool:

        jobs = []
        for year in years:
            ypath = catalog_path + str(year)

            for nfile in glob(ypath + '/*.nordic'):
                jobs.append(
                    pool.apply_async(
                        read_and_filter_bulletin,
                        (
                            nfile,
                            sites,
                            min_magnitude
                        )
                    )
                )
        
        for job in jobs:
            catalog += job.get()
    
    # Save catalog
    print(f'Retrieved {len(catalog)} events')
    
    return catalog
    

def save_event(event, output_dir, head_start, length):

    site = get_site_name(event)
    dist, _, _ = gps2dist_azimuth(
        ARCES_LAT, ARCES_LON,
        event.origins[0].latitude, event.origins[0].longitude
    )
    travel_time = dist/(P_VEL*1000)

    starttime = event.origins[0].time + travel_time - head_start
    stream = client.get_waveforms(
        'ARA*,ARB*,ARC*,ARD*',
        'BH*',
        starttime,
        starttime + length
    )

    # Check data integrity
    # If >= 3 sec is masked, remove the trace -- if less, fill with zeros
    masked_traces = []
    max_masked_length = 3.0 * stream.traces[0].stats.sampling_rate  # Max 3 sec 
    for tr in stream.traces:
        if isinstance(tr.data, np.ma.masked_array):
            masked_len = np.sum(tr.data.mask)
            if masked_len >= max_masked_length:
                masked_traces.append(tr)
            else:
                tr.data = tr.data.filled(0.0)
    
    max_masked_traces = 6
    if len(masked_traces) > max_masked_traces:
        print(event.origins[0].time, ':', len(masked_traces), 'masked traces, skipping')
        return False
    else:
        for trm in masked_traces:
            stream.remove(trm)

    outname = os.path.join(output_dir, site, event.origins[0].time.__str__()) + '.pkl'
    with open(outname, 'wb') as fout:
        pickle.dump(stream, fout)
    
    return True


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    ARCES_LAT, ARCES_LON = 69.53, 25.51
    P_VEL = 6.75
    HEAD_START = 30
    EVENT_LENGTH = 120
    client = Client()

    print('Compiling catalog...')
    years = [2015, 2016, 2017, 2018]
    min_magnitude = 1.0
    sites = ['KIRUNA', 'ZAPOLJARNY', 'SUURIKUUSIKKO', 'KEVITSA', 'KIROVSK']
    cat = create_event_list(years, min_magnitude, sites)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    for site in sites:
        sitepath = os.path.join(args.output_dir, site)
        if not os.path.isdir(sitepath):
            os.mkdir(sitepath)

    print('Saving events...')

    with Pool(processes=6) as pool:

        for event in cat.events:
            pool.apply(save_event, (event, args.output_dir, HEAD_START, EVENT_LENGTH))
        
    print('Done.')


    """
    i = 0
    for event in cat:

        site = get_site_name(event)
        dist, _, _ = gps2dist_azimuth(
            ARCES_LAT, ARCES_LON,
            event.origins[0].latitude, event.origins[0].longitude
        )
        travel_time = dist/(P_VEL*1000)
        #print('travel time:', travel_time)

        starttime = event.origins[0].time + travel_time - HEAD_START
        stream = client.get_waveforms(
            'ARA*,ARB*,ARC*,ARD*',
            'BHZ',
            starttime,
            starttime + EVENT_LENGTH
        )
        #stream.detrend('demean')
        #stream.taper(0.05)
        #stream.filter('bandpass', freqmin=3, freqmax=8)
        #stream.plot()

        outname = os.path.join(args.output_dir, site, event.origins[0].time.__str__()) + '.pkl'
        with open(outname, 'wb') as fout:
            pickle.dump(stream, fout)

        i += 1
        if i % 100 == 0:
            print(f'Saved {i} events')
    """




