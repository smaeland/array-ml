import pathlib
import random
from glob import iglob
from sys import path
import numpy as np
import h5py
from numpy.lib.function_base import append
import tensorflow as tf

"""
Utility function for train scripts
"""

class PreprocessedSteadGenerator(object):
    """
    Generator for sampling full STEAD events from a file of preprocessed data
    """

    def __call__(self, subset, test_split, seed):
        
        if isinstance(subset, bytes):
            subset = subset.decode()

        assert subset in ['train', 'test']
        assert 0.0 <= test_split <= 1.0

        datafile = '../data/STEAD/stead_noise_signal_mixed.h5'
        with h5py.File(datafile, 'r') as hin:
            
            group = hin.get('data')
            keys = list(group.keys())
            
            # Shuffle predictably
            keys.sort()
            random.seed(seed)
            random.shuffle(keys)

            pivot = round(len(keys) * test_split)
            
            if subset == 'test':
                keys = keys[:pivot]
            else:
                keys = keys[pivot:]
            
            print(f'subset {subset}: {len(keys)} events')
            
            for key in keys:

                dataset = group.get(key)
                str_label = dataset.attrs['trace_category']
                if str_label == 'noise':
                    label = 0.0
                else:
                    label = 1.0

                data = dataset[:]

                yield (data, label)



class ArrayWaveformGenerator(object):

    def __call__(self, directory, subset, seed=None, validation_split=None):
        """
        This is called once per epoch
        """

        if isinstance(directory, bytes):
            directory = directory.decode()
        if isinstance(subset, bytes):
            subset = subset.decode()
            
        assert subset in ['training', 'validation', 'testing']

        if validation_split is not None or subset is not None:
            assert 0.0 < validation_split < 1.0
            assert seed is not None

        files = []
        class_names = []
        base_dir = pathlib.Path(directory)
        for class_dir in sorted(base_dir.glob('*')):
            if class_dir.is_dir():
                class_names.append(class_dir.name)
                for npy_file in class_dir.glob('*.npy'):
                    files.append(npy_file)


        nclasses = len(class_names)
        assert nclasses > 1, f'classes found: {class_names}'
        assert len(files) > 2, f'files found: {files}'

        # Shuffle predictably
        files.sort()
        if seed is not None:
            random.seed(seed)        

        if subset in ['training', 'validation']:

            random.shuffle(files)
            pivot = round(len(files) * validation_split)
            
            if subset == 'validation':
                files = files[:pivot]
            else:
                files = files[pivot:]


        print(f'subset {subset}: {len(files)} events')

        for filename in files:

            label_name = filename.parent.name
            label = class_names.index(label_name)
            label = tf.keras.utils.to_categorical(label, num_classes=nclasses)

            data = np.load(filename)

            yield (data, label)


def save_numpy_arrays(input_path, output_path):
    """
    Convert pickle files to numpy for faster training
    """

    for class_dir in iglob(input_path + '/*'):

        output_dir = os.path.join(output_path, class_dir.split('/')[-1])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for filepath in iglob(class_dir + '/*.pkl'):

            with open(filepath, 'rb') as fin:
                stream = pickle.load(fin)
                stream = check_and_fill_missing_channels(stream, expected_channels)
                stream = cut_to_length(stream)
                stream = preprocess(stream)
                #stream.plot()
                data = stream_to_numpy(stream)

                outname = filepath.split('/')[-1]
                outname.replace('.pkl', '.npy')
                outpath = os.path.join(output_dir, outname)
                np.save(outpath, data)

    print('Done.')


def create_train_test_split_links(input_dir, output_dir, test_size: float):
    """
    Create train/test directory with symbolic links to original files
    """

    outdir_train = os.path.join(output_dir, 'train')
    if not os.path.exists(outdir_train):
        os.mkdir(outdir_train)

    outdir_test = os.path.join(output_dir, 'test')
    if not os.path.exists(outdir_test):
        os.mkdir(outdir_test)
    

    for class_dir in iglob(input_dir + '/*'):
        
        classname = class_dir.split('/')[-1]
        outdir_class_train = os.path.join(outdir_train, classname)
        outdir_class_test = os.path.join(outdir_test, classname)
        if not os.path.exists(outdir_class_train):
            os.mkdir(outdir_class_train)
        if not os.path.exists(outdir_class_test):
            os.mkdir(outdir_class_test)
        
        # Do each  class separately for proper stratification
        for filepath in iglob(class_dir + '/*.npy'):

            filename = filepath.split('/')[-1]
            if np.random.uniform() < test_size:
                os.symlink(filepath, os.path.join(outdir_class_test, filename))
            else:
                os.symlink(filepath, os.path.join(outdir_class_train, filename))

    print('Done.')

