"""
PhaseNet-type model, predicting P and S pick times, trained on STEAD data
"""
from pathlib import Path
import random
import platform
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.training.tracking.util import streaming_restore

def truncated_gauss(num_pts: int, loc: float, sigma: float):

    n = np.arange(0, num_pts) - loc
    w = np.exp(-(n*n) / (2*sigma*sigma))
    w[np.where(w < 1.03-3)] = 0.0 # Truncate low values

    return w


def target_traces_signal(length, p_pos, s_pos):

    p_trace = truncated_gauss(length, p_pos, 20)
    s_trace = truncated_gauss(length, s_pos, 20)
    n_trace = np.ones(shape=length) - p_trace - s_trace

    return np.vstack((p_trace, s_trace, n_trace)).T
    #return np.vstack((p_trace, s_trace)).T


def target_traces_noise(length):

    return np.vstack((
        np.zeros(length),
        np.zeros(length),
        np.ones(length)
    )).T


def plot_traces(data, target, pred=None):

    #assert data.shape == target.shape == (6000, 2)
    #if pred is not None:
    #    assert pred.shape == (6000, 2)

    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True)
    x = np.arange(0, data.shape[0]) # / SAMPLING_RATE
    ax1.plot(x, data[:, 0])
    ax2.plot(x, data[:, 1])
    ax3.plot(x, data[:, 2])
    ax4.plot(x, target)
    
    if pred is not None:
        ax5.plot(x, pred)

    plt.tight_layout()
    plt.show()


def preprocess_traces(data, remove_start=0):

    # Normalise
    newdata = data / np.max(np.abs(data))

    newdata = newdata[remove_start:, :]

    return newdata



class H5Generator(object):

    # TODO resampling, QC

    def __call__(self, filename):

        i = 0

        remove_start_pts = 60

        with h5py.File(filename, 'r') as fin:

            group = fin.get('data')

            
            # Shuffle predictably
            #keys = list(group.keys())
            #keys.sort()
            #random.seed(seed)
            #random.shuffle(keys)

            #for key in keys:
            for key in group.keys():
            
                dataset = group.get(key)
                tracelength = dataset.shape[0]
                cat = dataset.attrs['trace_category']
                rtype = dataset.attrs['receiver_type']
                if rtype not in ['BH', 'HH']:
                    continue
                
                if cat == 'earthquake_local':
                    p_start = dataset.attrs['p_arrival_sample']
                    s_start = dataset.attrs['s_arrival_sample']
                    target = target_traces_signal(tracelength-remove_start_pts, p_start-remove_start_pts, s_start-remove_start_pts)

                elif cat == 'noise':
                    target = target_traces_noise(tracelength-remove_start_pts)
                
                else:
                    continue
                
                if i >= 10000:
                    return

                data = preprocess_traces(dataset[:], remove_start_pts)
                yield (data, target)
                i += 1

    

def create_model(input_shape, n_classes):

    m = tf.keras.Sequential()

    # Input
    m.add(tf.keras.layers.Conv1D(3, kernel_size=7, input_shape=input_shape, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # Deep 1
    m.add(tf.keras.layers.Conv1D(10, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # Output
    m.add(tf.keras.layers.Conv1DTranspose(n_classes, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.Activation('softmax'))

    print(m.summary())

    return m


def phasenet_model(input_shape, n_classes):

    m = tf.keras.Sequential()

    # -> 8x3001
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, input_shape=input_shape, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 8x3001
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))
    
    # -> 8x751
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, strides=4, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 11x751
    m.add(tf.keras.layers.Conv1D(11, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))
    
    # -> 11x188
    m.add(tf.keras.layers.Conv1D(11, kernel_size=7, strides=4, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x188
    m.add(tf.keras.layers.Conv1D(16, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x47
    m.add(tf.keras.layers.Conv1D(16, kernel_size=7, strides=3, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x47
    m.add(tf.keras.layers.Conv1D(22, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x12
    #m.add(tf.keras.layers.Conv1D(22, kernel_size=7, strides=4, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    #m.add(tf.keras.layers.Activation('relu'))

    # -> 32x12
    #m.add(tf.keras.layers.Conv1D(32, kernel_size=7, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    #m.add(tf.keras.layers.Activation('relu'))

    # -> 44x47
    m.add(tf.keras.layers.Conv1DTranspose(44, kernel_size=7, strides=3, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x47
    m.add(tf.keras.layers.Conv1D(22, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 32x188
    m.add(tf.keras.layers.Conv1DTranspose(32, kernel_size=7, strides=4, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x188
    m.add(tf.keras.layers.Conv1D(16, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x751
    #m.add(tf.keras.layers.Conv1DTranspose(22, kernel_size=7, strides=4, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    #m.add(tf.keras.layers.Activation('relu'))

    # -> 11x751
    m.add(tf.keras.layers.Conv1D(11, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x3001
    m.add(tf.keras.layers.Conv1DTranspose(16, kernel_size=7, strides=4, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 8x3001
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 3x3001
    m.add(tf.keras.layers.Conv1D(n_classes, kernel_size=7, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('softmax'))

    print(m.summary())

    return m



def phasenet_model_40Hz(input_shape, n_classes):

    m = tf.keras.Sequential()

    # -> 8x3001
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, input_shape=input_shape, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 8x3001
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))
    
    # -> 8x751
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, strides=4, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 11x751
    m.add(tf.keras.layers.Conv1D(11, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))
    
    # -> 11x188
    m.add(tf.keras.layers.Conv1D(11, kernel_size=7, strides=3, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x188
    m.add(tf.keras.layers.Conv1D(16, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x47
    m.add(tf.keras.layers.Conv1D(16, kernel_size=7, strides=3, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x47
    m.add(tf.keras.layers.Conv1D(22, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x12
    #m.add(tf.keras.layers.Conv1D(22, kernel_size=7, strides=4, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    #m.add(tf.keras.layers.Activation('relu'))

    # -> 32x12
    #m.add(tf.keras.layers.Conv1D(32, kernel_size=7, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    #m.add(tf.keras.layers.Activation('relu'))

    # -> 44x47
    m.add(tf.keras.layers.Conv1DTranspose(44, kernel_size=7, strides=3, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x47
    m.add(tf.keras.layers.Conv1D(22, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 32x188
    m.add(tf.keras.layers.Conv1DTranspose(32, kernel_size=7, strides=3, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x188
    m.add(tf.keras.layers.Conv1D(16, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 22x751
    #m.add(tf.keras.layers.Conv1DTranspose(22, kernel_size=7, strides=4, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    #m.add(tf.keras.layers.Activation('relu'))

    # -> 11x751
    m.add(tf.keras.layers.Conv1D(11, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 16x3001
    m.add(tf.keras.layers.Conv1DTranspose(16, kernel_size=7, strides=4, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 8x3001
    m.add(tf.keras.layers.Conv1D(8, kernel_size=7, padding='same'))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('relu'))

    # -> 3x3001
    m.add(tf.keras.layers.Conv1D(n_classes, kernel_size=7, padding='same'))
    #m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.Activation('softmax'))

    print(m.summary())

    return m



if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if platform.node() == 'PC337':
        data_path = Path(r'C:\Users\steffen\projects\array-ml\data\STEAD')
    elif platform.node() == 'metis.norsar.no':
        data_path = '/nobackup/steffen/STEAD/chunk2.hdf5'


    data_shape = (2340, 3)

    stead_files = [
        'chunk1_resampled_40Hz.h5',
        'chunk2_resampled_40Hz.h5',
        'chunk3_resampled_40Hz.h5',
        'chunk4_resampled_40Hz.h5',
        'chunk5_resampled_40Hz.h5',
    ]
    stead_files = [str(data_path / f) for f in stead_files]
    ds = tf.data.Dataset.from_tensor_slices(stead_files)

    test_ds = ds.interleave(
        lambda filename: tf.data.Dataset.from_generator(
            H5Generator(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(data_shape), tf.TensorShape(data_shape)),
            args=(filename,)
        ),
        cycle_length=len(stead_files),
        block_length=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    )
    test_ds = test_ds.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

    val_ds_file = str(data_path/'chunk6_resampled_40Hz.h5')
    val_ds = tf.data.Dataset.from_generator(
        H5Generator(),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(data_shape), tf.TensorShape(data_shape)),
        args=(val_ds_file,)
    )
    val_ds = val_ds.batch(16)

    # Train
    model = phasenet_model_40Hz(data_shape, 3)
    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy', 'mean_squared_error'])
    
    model.fit(
        test_ds,
        validation_data=val_ds,
        epochs=30,
        verbose=2
    )

    model.save('model-phasenet-40Hz-jun28')
    

    # Test loading
    # model = tf.keras.models.load_model('model-phasenet-40Hz-jun28')
    # for X, Y in val_ds.take(5):
    #     pred = model.predict(X)
    #     for i in range(3): #X.shape[0]):
    #         plot_traces(X[i], Y[i], pred[i])


