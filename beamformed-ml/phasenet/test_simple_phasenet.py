import numpy as np
import scipy.signal
from obspy import Stream, Trace
import matplotlib.pyplot as plt
import tensorflow as tf

SAMPLING_RATE = 40.0


def synthetic_sample(plot=False):

    npts = 400

    x = np.arange(0, npts) / SAMPLING_RATE
    y = scipy.signal.gausspulse(x, fc=5, bw=0.2)
    y = np.roll(y, shift=npts//3)

    t = scipy.signal.windows.gaussian(npts, std=5.1)
    t = np.roll(t, shift=(-npts//2 + npts//3))
    t[np.where(t < 1.03-4)] = 0.0 # Truncate low values

    rnd_shift = np.random.randint(-100, 100)
    y = np.roll(y, shift=rnd_shift)
    t = np.roll(t, shift=rnd_shift)
    n = 1.0 - t

    y += np.random.normal(scale=0.1, size=npts)


    if plot:
        start = (npts//3 + rnd_shift - 1)/SAMPLING_RATE
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.plot(x, y)
        ax1.axvline(start, linestyle='dotted')
        ax2.plot(x, t)
        ax2.plot(x, n, color='red')
        ax2.axvline(start, linestyle='dotted')
        plt.show()

    return y.astype(np.float32), t.astype(np.float32)


def create_dataset(num):

    data = []
    targets = []

    for _ in range(num):
        y, t = synthetic_sample()
        data.append([y])
        targets.append([t])
    
    data = np.transpose(np.array(data), axes=[0, 2, 1])
    targets = np.transpose(np.array(targets), axes=[0, 2, 1])

    return data, targets



def create_model(input_shape):

    print('input_shape:', input_shape)

    n_classes = input_shape[-1]
    
    model = tf.keras.Sequential()

    # Input
    model.add(tf.keras.layers.Conv1D(3, kernel_size=8, input_shape=input_shape, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Deep 1
    model.add(tf.keras.layers.Conv1D(3, kernel_size=8, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Deep 2
    #model.add(tf.keras.layers.Conv1D(3, kernel_size=8, padding='same'))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Activation('relu'))

    # Output
    model.add(tf.keras.layers.Conv1DTranspose(n_classes, kernel_size=8, padding='same'))
    model.add(tf.keras.layers.Activation('sigmoid'))

    print(model.summary())

    return model



if __name__ == '__main__':

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    X, Y = create_dataset(5000)
    print('X.shape:', X.shape)
    print('Y.shape:', Y.shape)
    #create_model((400, 2))

    m = create_model(X.shape[1:])
    m.compile(loss='binary_crossentropy', metrics=['binary_accuracy', 'mean_squared_error'])
    m.fit(
        X, Y, 
        epochs=10,
        verbose=1
    )

    X_test, Y_test = create_dataset(1)
    pred = m.predict(X_test)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    x = np.arange(0, 400) / SAMPLING_RATE
    ax1.plot(x, X_test[0])
    ax2.plot(x, Y_test[0])
    ax2.plot(x, pred[0], color='red')

    X_noise = np.random.normal(scale=0.1, size=400)
    X_noise = np.expand_dims(X_noise, axis=0)
    X_noise = np.expand_dims(X_noise, axis=-1)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    x = np.arange(0, 400) / SAMPLING_RATE
    ax1.plot(x, X_noise[0])
    ax2.plot(x, m.predict(X_noise)[0], color='red')
    plt.show()
