"""
Train CNNs on STEAD data
"""

import os.path
import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from obspy import Stream, Trace


def select_events(filename: str, maxevents=None):
    """
    Select local/regional events from csv files
    """
    data = []

    with h5py.File(filename, 'r') as fin:

        group = fin.get('data')
        for key in group.keys():
            
            dataset = group.get(key)
            cat = dataset.attrs['trace_category']
            rtype = dataset.attrs['receiver_type']
            if cat == 'earthquake_local' and rtype in ['BH', 'HH']:
                data.append(dataset[()])
            
            if maxevents is not None and len(data) >= maxevents:
                break
    
    return np.array(data)


def get_arrival_segments(filename: str, maxevents=None):
    """
    TODO: preprocessing
    """

    window_length_sec = 5
    sampling_rate = 40.0
    window_length_npts = int(window_length_sec * sampling_rate)

    p_wave_data, s_wave_data = [], []

    with h5py.File(filename, 'r') as fin:

        group = fin.get('data')
        for key in group.keys():
            
            dataset = group.get(key)
            cat = dataset.attrs['trace_category']
            rtype = dataset.attrs['receiver_type']
            if rtype not in ['BH', 'HH']:
                continue
            
            # Local earthquake
            if cat == 'earthquake_local':

                p_start = dataset.attrs['p_arrival_sample']
                s_start = dataset.attrs['s_arrival_sample']

            # Noise
            elif cat == 'noise':
                
                p_start = np.random.uniform(low=0, high=dataset.shape[0]//2)
                s_start = np.random.uniform(low=p_start + window_length_npts, high=dataset.shape[0] - window_length_npts)
            
            #stream = Stream

            p_window_start = int(p_start - window_length_npts//2)
            p_window_end = int(p_start + window_length_npts//2)
            p_window = dataset[p_window_start:p_window_end, :]
            if p_window.shape == (window_length_npts, 3):
                p_wave_data.append(p_window)
            else:
                print('P window shape:', p_window.shape, '- skipping')

            s_window_start = int(s_start - window_length_npts//2)
            s_window_end = int(s_start + window_length_npts//2)
            s_window = dataset[s_window_start:s_window_end, :]
            if s_window.shape == (window_length_npts, 3):
                s_wave_data.append(s_window)
            else:
                print('S window shape:', s_window.shape, '- skipping')


            if maxevents is not None and len(p_wave_data) >= maxevents:
                break
    
    return np.array(p_wave_data), np.array(s_wave_data)


def get_noise_segments(filename: str, maxevents=None):

    window_length_sec = 5
    sampling_rate = 40.0
    window_length_npts = int(window_length_sec * sampling_rate)

    data = []

    with h5py.File(filename, 'r') as fin:

        group = fin.get('data')
        for key in group.keys():
            
            dataset = group.get(key)
            cat = dataset.attrs['trace_category']
            rtype = dataset.attrs['receiver_type']
            if rtype not in ['BH', 'HH']:
                continue
            
            assert cat == 'noise'

            start_index, stop_index = 0, window_length_npts
            while stop_index < dataset.shape[0]:
                data.append(dataset[start_index: stop_index, :])
                start_index = stop_index
                stop_index += window_length_npts
            
                if maxevents is not None and len(data) >= maxevents:
                    return np.array(data)
        
    return np.array(data)
                






def create_model(input_shape, num_classes):

    c1_filters, c1_len = 10, 8
    c2_filters, c2_len = 20, 16
    c3_filters, c3_len = 30, 32

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv1D(
            c1_filters,
            kernel_size=c1_len,
            input_shape=input_shape
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.1))

    model.add(
        tf.keras.layers.Conv1D(c2_filters, kernel_size=c2_len)
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.1))

    model.add(
        tf.keras.layers.Conv1D(c3_filters, kernel_size=c3_len)
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.SpatialDropout1D(0.1))

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model





def train():
    
    data_path = '/nobackup/steffen/STEAD'
    X_pwave, X_swave = get_arrival_segments(os.path.join(data_path, 'chunk6.hdf5'), maxevents=1000)
    Y_pwave = np.ones(shape=X_pwave.shape[0])
    Y_swave = np.ones(shape=Y_pwave.shape[0]) + 1

    X_noise = get_noise_segments(os.path.join(data_path, 'chunk1.hdf5'), maxevents=1000)
    Y_noise = np.zeros(shape=X_noise.shape[0])

    X = np.vstack((X_pwave, X_swave, X_noise))
    Y = np.hstack((Y_pwave, Y_swave, Y_noise))
    Y = tf.keras.utils.to_categorical(Y)

    print('X.shape:', X.shape)
    print('Y.shape:', Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

    model = create_model(input_shape=X.shape[1:], num_classes=3)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=10,
        epochs=1,
        verbose=1
    )

    model.save('testmodel')



if __name__ == '__main__':

    train()