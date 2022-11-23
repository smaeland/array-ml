"""
Process STEAD data and write to smaller files
"""
from random import shuffle
from multiprocessing import Pool
import numpy as np
import h5py
from obspy import Stream, Trace

def process_data(data: np.ndarray):

    st = Stream()
    for i in range(data.shape[1]):
        st.append(
            Trace(data=data[:, i], header={
                'channel': str(i),
                'sampling_rate': 100.0
            })
        )
    
    st.detrend('demean')
    st.taper(max_percentage=None, max_length=2, type='cosine')
    st.filter('bandpass', freqmin=1.0, freqmax=20.0)
    st.resample(80.0)
    st.normalize(global_max=True)

    newdata = [tr.data.astype(np.float32) for tr in st.traces]
    newdata = np.transpose(np.array(newdata))

    return newdata
    



def write_h5_file(eventlist, input_data_files, output_filename):

    # Open all input files
    input_files = {}
    input_groups = {}
    for f in input_data_files:
        input_files[f] = h5py.File(f, 'r')
        input_groups[f] = input_files[f].get('data')
    
    # Output file
    outfile = h5py.File(output_filename, 'w')
    outgroup = outfile.create_group('data')

    # Copy events
    for filename, eventname in eventlist:
        
        assert eventname in input_groups[filename].keys()
        
        dataset_in = input_groups[filename].get(eventname)
        dataset_out = outgroup.create_dataset(
            eventname,
            data=process_data(dataset_in[:])
        )
        for key, value in dataset_in.attrs.items():
            dataset_out.attrs[key] = value

    for f in input_files:
        input_files[f].close()
    outfile.close()

    return output_filename



def convert():

    basepath = '/nobackup/steffen/STEAD/'
    outpath = '/nobackup/steffen/STEAD/redistributed/'
    csv_files = [
        'chunk1.csv',
        'chunk2.csv',
        'chunk3.csv',
        'chunk4.csv',
        'chunk5.csv',
        'chunk6.csv',
    ]
    csv_files = [basepath + f for f in csv_files]
    data_files = [f.replace('.csv', '.hdf5') for f in csv_files]
    entries = []
    
    # Get list of entries that pass requirements
    for csvfile in csv_files:

        with open(csvfile, 'r') as fin:
            for line in fin:
                line = line.strip().split(',')
                receiver_type = line[2]
                trace_category = line[-2]
                trace_name = line[-1]

                if (receiver_type in ['BH', 'HH'] and
                    trace_category in ['earthquake_local', 'noise']
                ):  
                    datafile = csvfile.replace('.csv', '.hdf5')
                    entries.append((datafile, trace_name))

    print('len(entries):', len(entries))

    shuffle(entries)

    # Write to new files
    num_events_per_file = 10000

    # Sequential
    #while len(entries) > 0:
    #    to_process = entries[:num_events_per_file]
    #    outname = f'datafile-test-{i}'
    #    write_h5_file(to_process, data_files, outname)
    #    entries = entries[num_events_per_file:]
    #    i += 1

    # Parallel
    results = []
    with Pool(processes=8) as pool:

        start = 0
        i = 0
        while start < len(entries):
            i += 1
            outname = f'{outpath}datafile-chunk-{i}.h5'
            stop = start + num_events_per_file
            results.append(
                pool.apply_async(
                    write_h5_file,
                    (entries[start: stop], data_files, outname)
                )
            )
            start = stop

        for res in results:
            print('Wrote', res.get())


if __name__ == '__main__':
    convert()