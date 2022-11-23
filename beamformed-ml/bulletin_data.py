from obspy import UTCDateTime
from sqlalchemy import create_engine, text
from seismonpy.norsardb import Client
import matplotlib.pyplot as plt


def norsar_bulletin():
    
    engine = create_engine('oracle://seisana:seismon@abbor/ora11g')
    conn = engine.connect()

    qry_assocs = text('SELECT arrival.arid, arrival.time, arrival.iphase '
                      'FROM arrival '
                      'INNER JOIN assoc ON arrival.arid=assoc.arid '
                      'WHERE arrival.sta = :station AND arrival.time > :start')

    arrival_times = []

    starttime = UTCDateTime('2014-09-01T00:00:00').timestamp

    assocs = conn.execute(qry_assocs, station='ARCES', start=starttime)

    for assoc in assocs:

        if assoc is None:
            print('Error: No arrivals')
            continue

        arid = assoc[0]

        qry_arrivals = text('SELECT time, iphase FROM arrival WHERE arid = :arid')
        arrivals = conn.execute(qry_arrivals, arid=arid)
        arrivals = arrivals.fetchall()

        if len(arrivals) != 1:
            print(f'Error: {len(arrivals)} arrivals for arid {arid}')
            continue

        arrivaltime = UTCDateTime(float(arrivals[0][0]))
        phase = str(arrivals[0][1])

        if phase.upper().startswith('P'):
            arrival_times.append(arrivaltime)
            #create_spectrogram(arrivaltime, outpath)

    return arrival_times


def test_trig():

    from obspy.signal.trigger import plot_trigger
    from obspy.signal.trigger import z_detect

    st = Client().get_waveforms('ARA0', 'HHZ', UTCDateTime('2021-01-05T08:33:00'), UTCDateTime('2021-01-05T08:36:00'))
    st.detrend('demean')
    st.taper(0.03)
    st.filter('bandpass', freqmin=3, freqmax=6)
    #st.plot()
    trace = st.traces[0]

    df = trace.stats.sampling_rate
    #cft = z_detect(trace.data, int(10 * df))
    #plot_trigger(trace, cft, -0.4, -0.3)

    from obspy.signal.trigger import recursive_sta_lta
    cft = recursive_sta_lta(trace.data, int(5 * df), int(10 * df))
    plot_trigger(trace, cft, 1.5, 1.0)



if __name__ == '__main__':

    #at = norsar_bulletin()
    #print('num arrivals:', len(at))

    test_trig()



