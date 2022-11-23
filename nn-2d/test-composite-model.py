import numpy as np
import tensorflow as tf
from pretrain_on_stead import PreprocessedSteadGenerator



if __name__ == '__main__':

    data_shape = (2400, 3)

    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='same', activation='relu'))
    model1.add(tf.keras.layers.Conv1D(filters=10, kernel_size=8, padding='same', activation='relu'))
    model1.add(tf.keras.layers.Conv1D(filters=10, kernel_size=12, padding='same', activation='relu'))


    input_layer = tf.keras.Input(shape=data_shape)
    convs = model1(input_layer)
    gap = tf.keras.layers.GlobalAveragePooling1D()(convs)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(gap)
    
    model2 = tf.keras.Model(inputs=input_layer, outputs=out)


    train_ds = tf.data.Dataset.from_generator(
        PreprocessedSteadGenerator(),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(data_shape), tf.TensorShape(())),
        args=()
    )

    train_ds = train_ds.batch(12).prefetch(tf.data.experimental.AUTOTUNE)


    model2.compile(loss='binary_crossentropy', metrics=['binary_accuracy'])
    print(model2.summary())

    #model2.fit(
    #    train_ds,
    #    epochs=1,
    #    verbose=1
    #)


    

