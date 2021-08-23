from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import signal
import h5py
import matplotlib.pyplot as plt


def resample(filename_in, filename_out, num_samples):

    fin = h5py.File(filename_in, 'r')
    fout = h5py.File(filename_out, 'w')

    group_in = fin.get('data')
    group_out = fout.create_group('data')

    for key in group_in.keys():

        dataset_in = group_in.get(key)

        data_in = dataset_in[:]
        data_out = np.zeros(shape=(num_samples, data_in.shape[1]))

        for i in range(data_in.shape[1]):
            data_out[:, i] = signal.resample(data_in[:, i], num_samples)
        
        dataset_out = group_out.create_dataset(key, data=data_out, dtype=np.float32)

        # Set attributes
        for key, value in dataset_in.attrs.items():

            if key == 'p_arrival_sample' or key == 's_arrival_sample':
                if value is not None and not isinstance(value, str):    # for noise samples
                    value = (value / data_in.shape[0]) * num_samples
                    value = int(value)

            #print(key, value)
            dataset_out.attrs[key] = value

        dataset_out.attrs['sampling_rate'] = (num_samples / data_in.shape[0]) * 100.0

        plot = False
        if plot:
            _, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6)
            x1 = np.arange(0, data_in.shape[0]) # / SAMPLING_RATE
            ax1.plot(x1, data_in[:, 0])
            ax2.plot(x1, data_in[:, 1])
            ax3.plot(x1, data_in[:, 2])
            ax1.axvline(dataset_in.attrs['p_arrival_sample'])
            ax1.axvline(dataset_in.attrs['s_arrival_sample'])
            x2 = np.arange(0, data_out.shape[0]) # / SAMPLING_RATE
            ax4.plot(x2, data_out[:, 0])
            ax5.plot(x2, data_out[:, 1])
            ax6.plot(x2, data_out[:, 2])
            ax4.axvline(dataset_out.attrs['p_arrival_sample'])
            ax4.axvline(dataset_out.attrs['s_arrival_sample'])
            plt.tight_layout()
            plt.show()

    fin.close()
    fout.close()


if __name__ == '__main__':

    base_path = Path(r'C:\Users\steffen\projects\array-ml\data\STEAD')
    files = [
        'chunk1.hdf5',
        'chunk2.hdf5',
        'chunk3.hdf5',
        'chunk4.hdf5',
        'chunk5.hdf5',
        'chunk6.hdf5',
    ]

    with Pool(6) as pool:

        jobs = []
        for f in files:
            jobs.append(
                pool.apply_async(
                    resample,
                    (
                        base_path / f,
                        base_path / f.replace('.hdf5', '_resampled_40Hz.h5'),
                        2400
                    )
                )
            )
        
        for j in jobs:
            j.get()
        
    print('Done.')
