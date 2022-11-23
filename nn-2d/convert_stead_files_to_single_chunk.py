import random
import numpy as np
import h5py
from obspy import Stream, Trace


"""
Make a single file with noise and earthquakes to be used for pretraining
"""



def merge_noise_earthquake_events():

    noise_file = '/nobackup/steffen/STEAD/chunk1.hdf5'
    signal_file = '/nobackup/steffen/STEAD/chunk2.hdf5'
    output_file = 'stead_noise_signal_mixed.h5'

    keys = []
    
    nf = h5py.File(noise_file, 'r')
    ng = nf.get('data')
    for key in ng.keys():
        keys.append((str(key), 'noise'))
    
    sf = h5py.File(signal_file, 'r')
    sg = sf.get('data')
    for key in sg.keys():
        keys.append((str(key), 'signal'))
    
    random.shuffle(keys)

    outfile = h5py.File(output_file, 'w')
    outgroup = outfile.create_group('data')

    for key, ktype in keys:

        if ktype == 'signal':
            dataset = sg.get(key)
            label = 1
        else:
            dataset = ng.get(key)
            label = 0
        
        rtype = dataset.attrs['receiver_type']
        if rtype not in ['BH', 'HH']:
            continue

        #assert channels_sorted_correctly
        # Channels are E-N-Z (sorted)

        stream = Stream()
        for i in range(dataset.shape[1]):
            stream.append(
                Trace(
                    data=dataset[:, 0],
                    header={
                        'station': 'STA',
                        'channel': 'ENZ'[i],
                        'sampling_rate': 100.0
                    }
                )
            )


        stream.resample(40.0)
        stream.detrend('demean')
        stream.taper(max_percentage=None, max_length=3)
        stream.filter('highpass', freq=1.5)
        stream.normalize(global_max=True)

        data = []
        for trace in stream.traces:
            data.append(trace.data)
            
        data = np.array(data).T

        dataset_out = outgroup.create_dataset(key, data=data)
    
        for key, value in dataset.attrs.items():
            dataset_out.attrs[key] = value
        dataset_out.attrs['sampling_rate'] = 40.0
        


    outfile.close()
    nf.close()
    sf.close()



if __name__ == '__main__':

    merge_noise_earthquake_events()