import pathlib
from glob import iglob
from argparse import ArgumentParser
import platform
import numpy as np
import tensorflow as tf
from obspy import Stream, Trace
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score
)
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.saving.save import load_model
from nn_utilities import PreprocessedSteadGenerator, ArrayWaveformGenerator



def create_3C_model(input_shape):

    model = tf.keras.Sequential()

    #model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='same', activation='relu'))#, input_shape=input_shape))
    #model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=8, padding='same', activation='relu'))
    #model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=12, padding='same', activation='relu'))

    model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=8, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv1D(filters=10, kernel_size=12, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    return model


def convert_input_shape(model, new_input_shape):
    """
    Change the input shape of the model to match the new time axis length
    """
    model._layers[0]._batch_input_shape = (None, ) + new_input_shape
    new_model = tf.keras.models.model_from_json(model.to_json())
    new_model.summary()
    for layer, new_layer in zip(model.layers,new_model.layers):
        new_layer.set_weights(layer.get_weights())
        print('old layer:', layer)
        print('new layer:', new_layer)

    return new_model


def create_2D_model(input_shape, model_3C, num_classes):

    num_channels = input_shape[-1]
    num_stations = num_channels // 3

    # Input layer
    input_layer = tf.keras.Input(shape=input_shape)

    # Layer to expand last dimension after 1D convolution model, so that
    # 1D convolution outputs can be concatenated to an image
    expanddim2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))


    # Function to extract E-N-Z triplets from full stream
    def extract_3C(inputs, cxs):
        """ cxs: Tuple(int, int) """
        return tf.keras.layers.Lambda(lambda x: x[:, :, cxs[0]:cxs[1]])(inputs)

    # Get 3C data streams and collect them in a list
    xs_3C = []
    col_indices = list(range(0, num_channels+1, 3))
    col_index_pairs = zip(col_indices, col_indices[1:]) # [(0,3), (3,6), ...]

    for idx_pair in col_index_pairs:
        xs_3C.append(extract_3C(input_layer, idx_pair))

    assert len(xs_3C) == num_stations

    # Apply the 3C convolution model to all 3C streams,
    # then expand last dimension
    for i in range(len(xs_3C)):
        xs_3C[i] = model_3C(xs_3C[i])
        xs_3C[i] = expanddim2(xs_3C[i])
    
    # Concatenate 1D convolution outputs to a 2D image
    concat = tf.keras.layers.Concatenate(axis=2)(xs_3C)

    # Apply 2D convolutions
    # Make sure that kernel size in the non-time dimension is exactly 
    # equal to number of stations
    #conv2d = tf.keras.layers.Conv2D(10, kernel_size=(8, num_stations), padding='valid', activation='relu')(concat)    ## padding!
    conv2d = tf.keras.layers.Conv2D(10, kernel_size=(5, num_stations), padding='valid')(concat)
    conv2d = tf.keras.layers.BatchNormalization()(conv2d)
    conv2d = tf.keras.layers.Activation('relu')(conv2d)

    conv2d = tf.keras.layers.Conv2D(10, kernel_size=(10, num_stations), padding='valid')(concat)
    conv2d = tf.keras.layers.BatchNormalization()(conv2d)
    conv2d = tf.keras.layers.Activation('relu')(conv2d)

    conv2d = tf.keras.layers.Conv2D(10, kernel_size=(15, num_stations), padding='valid')(concat)
    conv2d = tf.keras.layers.BatchNormalization()(conv2d)
    conv2d = tf.keras.layers.Activation('relu')(conv2d)

    # Take global average, then classify
    gap = tf.keras.layers.GlobalAveragePooling2D()(conv2d)
    output_activ = 'sigmoid' if num_classes == 2 else 'softmax'
    output_layer = tf.keras.layers.Dense(num_classes, activation=output_activ)(gap)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

    return model





if __name__ == '__main__':

    parser = ArgumentParser('Train 2D CNN for arrays')
    parser.add_argument('-pt', '--pretrain_on_stead_data', action='store_true')
    parser.add_argument('-tp', '--test_pretrained_model', action='store_true')
    parser.add_argument('-t', '--train_on_array_data', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-cf', '--convert_files', action='store_true')
    parser.add_argument('-sp', '--train_test_split_links', action='store_true')
    args = parser.parse_args()

    stead_data_shape = (2400, 3)


    # -------------------------------------------------------------------------
    if args.convert_files:
        inpath = '/nobackup/steffen/array-ml-data-may2021/pickle'
        outpath = '/nobackup/steffen/array-ml-data-may2021/numpy'

        from nn_utilities import save_numpy_arrays    
        save_numpy_arrays(inpath, outpath)


    # -------------------------------------------------------------------------
    if args.train_test_split_links:
        inpath = '/nobackup/steffen/array-ml-data-may2021/numpy'
        outpath = '/nobackup/steffen/array-ml-data-may2021'

        from nn_utilities import create_train_test_split_links
        create_train_test_split_links(inpath, outpath, test_size=0.3)


    # -------------------------------------------------------------------------
    if args.pretrain_on_stead_data:
        
        print('Pretraining 1D convolution model...')

        # Datasets
        train_ds = tf.data.Dataset.from_generator(
            PreprocessedSteadGenerator(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(stead_data_shape), tf.TensorShape(())),
            args=('train', 0.3, 123)
        )
        val_ds = tf.data.Dataset.from_generator(
            PreprocessedSteadGenerator(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(stead_data_shape), tf.TensorShape(())),
            args=('test', 0.3, 123)
        )

        train_ds = train_ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

        # Compose model of input, conv layers, classification layer
        # then save only the conv part
        input_layer = tf.keras.Input(shape=stead_data_shape)
        stead_model = create_3C_model(stead_data_shape)
        x = stead_model(input_layer)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ]
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=12,
            verbose=2
        )
        print('Train metrics:')
        print(model.evaluate(train_ds, verbose=0))
        print('Validation metrics:')
        print(model.evaluate(val_ds, verbose=0))

        model_file = 'stead-model-jun14'
        stead_model.save(model_file)

        
    # -------------------------------------------------------------------------
    if args.test_pretrained_model:

        # Test that the saved model can be loaded
        print('Testing saved model...')
        loaded_conv_model = tf.keras.models.load_model('stead-model-jun07')
        for layer in loaded_conv_model.layers:
            layer.trainable = False

        input_layer = tf.keras.Input(shape=stead_data_shape)
        xx = loaded_conv_model(input_layer)
        xx = tf.keras.layers.GlobalAveragePooling1D()(xx)
        xx = tf.keras.layers.Dense(1, activation='sigmoid')(xx)
        loaded_model = tf.keras.Model(inputs=input_layer, outputs=xx)

        loaded_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ]
        )

        train_ds = tf.data.Dataset.from_generator(
            PreprocessedSteadGenerator(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(stead_data_shape), tf.TensorShape(())),
            args=('train', 0.3, 123)
        )
        train_ds = train_ds.batch(12).prefetch(tf.data.experimental.AUTOTUNE)

        loaded_model.fit(
            train_ds,
            epochs=1,
            verbose=1
        )
        print('Train metrics:')
        print(loaded_model.evaluate(train_ds))


    # -------------------------------------------------------------------------
    if args.train_on_array_data:

        if platform.node() == 'PC337':
            train_data_path = pathlib.Path(R'..\data\array-ml-data-may2021\train')
            test_data_path = pathlib.Path(R'..\data\array-ml-data-may2021\test')

        # Open an process a single file to get shapes
        ref_file = pathlib.Path.joinpath(train_data_path, 'KIRUNA').glob('*.npy').__next__()
        data = np.load(ref_file)
        data_shape = data.shape

        print('data_shape:', data_shape)
        nclasses = 5

        # Load pretrained conv part
        stead_model = tf.keras.models.load_model('stead-model-jun14')

        # Change the input shape, then fix the conv layers 
        data_shape_3C = (data_shape[0], 3)
        stead_model = convert_input_shape(stead_model, data_shape_3C)
        for layer in stead_model.layers:
            layer.trainable = False


        # Create full model
        model = create_2D_model(data_shape, stead_model, nclasses)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ]
        )

        print(model.summary())


        train_ds = tf.data.Dataset.from_generator(
            ArrayWaveformGenerator(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(data_shape), tf.TensorShape((nclasses, ))),
            args=(str(train_data_path), 'training', 10, 0.2)
        )
        val_ds = tf.data.Dataset.from_generator(
            ArrayWaveformGenerator(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(data_shape), tf.TensorShape((nclasses, ))),
            args=(str(train_data_path), 'validation', 10, 0.2)
        )

        train_ds = train_ds.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

        earlystop = tf.keras.callbacks.EarlyStopping(
            patience=5, verbose=1, restore_best_weights=True
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[earlystop],
            verbose=2
        )

        model.save('full-model-jun14')


    if args.evaluate:

        if platform.node() == 'PC337':
            train_data_path = pathlib.Path(R'..\data\array-ml-data-may2021\train')
            test_data_path = pathlib.Path(R'..\data\array-ml-data-may2021\test')

        
        data_shape = (2401, 48)
        nclasses = 5

        val_ds = tf.data.Dataset.from_generator(
            ArrayWaveformGenerator(),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape(data_shape), tf.TensorShape((nclasses, ))),
            args=(str(train_data_path), 'validation', 10, 0.2)
        )
        val_ds = val_ds.batch(16)

        model = tf.keras.models.load_model('full-model-jun14')

        print('Metrics:')
        print(model.evaluate(val_ds, verbose=0))

        preds = []
        targets= []
        for bx, by in val_ds:   # load batches
            
            bpreds = model.predict(bx)
            for p, y in zip(bpreds, by):
                targets.append(y)
                preds.append(p)

        preds = np.array(preds)
        targets = np.array(targets)

        preds_manyhot = np.argmax(preds, axis=1)
        targets_manyhot = np.argmax(targets, axis=1)

        print('accuracy:', accuracy_score(targets_manyhot, preds_manyhot))
        print('precision (micro):', precision_score(targets_manyhot, preds_manyhot, average='micro'))
        print('precision (macro):', precision_score(targets_manyhot, preds_manyhot, average='macro'))
        print('precision (weighted):', precision_score(targets_manyhot, preds_manyhot, average='weighted'))
        print('recall (micro):', recall_score(targets_manyhot, preds_manyhot, average='micro'))
        print('recall (macro):', recall_score(targets_manyhot, preds_manyhot, average='macro'))
        print('recall (weighted):', recall_score(targets_manyhot, preds_manyhot, average='weighted'))

        print('confusion matrix:')
        print(confusion_matrix(targets_manyhot, preds_manyhot))






