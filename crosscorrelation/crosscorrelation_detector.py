
import os.path
import sys
import pickle
from glob import iglob
import warnings
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Dict
import numpy as np
import tensorflow as tf
import scipy.signal
import scipy.cluster
from matplotlib import pyplot as plt
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.cross_correlation import correlation_detector
from obspy.geodetics import gps2dist_azimuth
from seismonpy.norsardb import Client
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extract_data import create_event_list


def preprocess(stream, edge_length):
    """
    Uniform preprocessing
    """
    out = Stream(stream)
    out.detrend('demean')
    out.taper(max_percentage=None, max_length=edge_length, type='cosine')
    out.filter('bandpass', freqmin=3, freqmax=8)
    st = stream.traces[0].stats.starttime
    nd = stream.traces[0].stats.endtime
    out.trim(st + edge_length, nd - edge_length)

    return out


def cluster_on_reference_station(site, ref_station='ARA0', ref_chan='BHZ'):
    """
    Find template events by clustering on reference station
    """

    if site == 'KIRUNA':
        p_vel = 6.65
        length = 65
        min_magnitude = 2.0
    elif site == 'KEVITSA':
        p_vel = 6.2
        length = 65
        min_magnitude = 1.3
    elif site == 'KIROVSK':
        p_vel = 7.0
        length = 80
        min_magnitude = 1.9
    elif site == 'SUURIKUUSIKKO':
        p_vel = 6.0
        length = 40
        min_magnitude = 0.8
    elif site == 'ZAPOLJARNY':
        p_vel = 6.0
        length = 50
        min_magnitude = 1.5
    else:
        raise RuntimeError(f'Not a valid site: {site}')


    print('Compiling catalog...')
    years = [2015, 2016, 2017, 2018]
    
    ARCES_LAT, ARCES_LON = 69.53, 25.51
    client = Client()

    cat = create_event_list(years, min_magnitude, [site])

    data = []
    labels = []

    for event in cat:

        dist, _, _ = gps2dist_azimuth(
        ARCES_LAT, ARCES_LON,
        event.origins[0].latitude, event.origins[0].longitude
        )
        travel_time = dist/(p_vel*1000)

        head_start = 2
        edge = 5
        starttime = event.origins[0].time + travel_time - head_start - edge
        stream = client.get_waveforms(ref_station, ref_chan, starttime, starttime + length + edge)

        try:
            stream = preprocess(stream, edge_length=edge)
        except IndexError as exc:
            print('event', event.origins[0].time, ':', exc)
            continue
        except NotImplementedError as exc:
            print('event', event.origins[0].time, ':', exc)
            continue
        
        #stream.plot()

        tracedata = stream.traces[0].data
        tracedata /= np.linalg.norm(tracedata)
        data.append(tracedata)

        label = str(event.origins[0].time) + ' ' + str(event.magnitudes[0].mag) + str(event.magnitudes[0].magnitude_type)
        labels.append(label)
    

    compute_dendrogram(data, labels)



def max_correlation(arr1, arr2):
    corr = scipy.signal.correlate(arr1, arr2, mode='full')
    return np.max(corr)


def compute_dendrogram(data, labels):

    # Compute correlation matrix
    correlation_threshold = 0.5
    n_obs = len(data)
    corr_matrix = np.zeros(shape=(n_obs, n_obs), dtype=np.float32)

    with Pool(processes=10) as pool:

        jobs = {}
        for i in range(n_obs):
            for j in range(n_obs):
                # Compute upper right triangle
                if j >= i:
                    jobs[(i, j)] = pool.apply_async(
                        max_correlation, (data[i], data[j])
                    )
        
        for i in range(n_obs):
            for j in range(n_obs):
                # Upper right triangle
                if j >= i:
                    maxcorr = jobs[(i, j)].get()
                    corr_matrix[i, j] = maxcorr

    for i in range(n_obs):
        for j in range(n_obs):
            # Lower left triangle
            if j < i:
                corr_matrix[i, j] = corr_matrix[j, i]

    # Zero the diagonal
    for i in range(n_obs):
        corr_matrix[i, i] = 0


    # Remove entries that poorly correlate with any other event
    max_per_column = np.max(corr_matrix, axis=0)
    above_thres = []
    above_thres_times = []
    for i in range(max_per_column.shape[0]):
        if max_per_column[i] > correlation_threshold:
            above_thres.append(i)
            above_thres_times.append(labels[i])
    
    print('Removing {} observations due to low correlation'.format(
        corr_matrix.shape[0] - len(above_thres)
    ))

    corr_matrix = corr_matrix[:, above_thres]
    corr_matrix = corr_matrix[above_thres, :]
    labels = above_thres_times
    
    # Compute distances
    for i in range(corr_matrix.shape[0]):
        corr_matrix[i, i] = 1   # reset diagonals to 1
    dist = 1.0 - corr_matrix
    #print(dist)
    dist = scipy.spatial.distance.squareform(dist)

    linkage = scipy.cluster.hierarchy.linkage(dist, method='single', optimal_ordering=True)
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, orientation='left', distance_sort='ascending', labels=labels)
    plt.show()




def predict_on_saved_files_OLD(
    file_path,
    template_starttime,
    template_endtime,
    stations='ARA*,ARB*,ARC*,ARD*',
    channels='BHZ',
    plot=False
):
    
    # Get template
    edge = 5
    template = Client().get_waveforms(
        stations, channels, template_starttime - edge, template_endtime + edge
    )
    template = preprocess(template, edge)

    template.plot()

    true_labels = []
    preds = []

    # Load files and process
    #for sitepath in iglob(file_path + '/*'):
    for sitepath in iglob(file_path + '/SUU*'):

        site = sitepath.split('/')[-1]
        for filename in iglob(sitepath + '/*.pkl'):

            with open(filename, 'rb') as fin:
                stream = pickle.load(fin)
            
            stream = preprocess(stream, edge)
            stream = stream.select(channel='*Z')

            print(site)
            detections, sims = correlation_detector(
                stream, template,
                heights=0.2, distance=10,
                plot=stream
            )
            print(detections)
            print(sims)
            


def get_template_stream(starttime, length):

    stations = 'ARA*,ARB*,ARC*,ARD*'
    channels = 'BHZ'
    edge = 5

    stream = Client().get_waveforms(
        stations, channels, starttime - edge, starttime + length + edge
    )
    stream = preprocess(stream, edge)

    return stream



def compute_similarities(input_file, templates, thresh, classlist, cat_label):

    with open(input_file, 'rb') as fin:
        stream = pickle.load(fin)
    
    stream = preprocess(stream, 3)
    event_preds = []

    for class_name in classlist:
        detections, _ = correlation_detector(
            stream,
            templates[class_name],
            heights=thresh,
            distance=10
        )
        sims = [0]
        for det in detections:
            sims.append(det['similarity'])
        
        event_preds.append(max(sims))
    
    # No detection
    if sum(event_preds) == 0:
        event_preds.append(1)
    else:
        event_preds.append(0)
    
    return (event_preds, cat_label)



def predict_parallel(template_streams: Dict[str, Stream], input_path: str, threshold: float):

    class_list = [d.split('/')[-1] for d in iglob(input_path + '/*')]
    class_list.sort()
    print('classes:', class_list + ['None'])

    true_labels = []
    preds = []
    jobs = []

    with Pool(processes=10) as pool:

        #i = 0
        for label_num, label_name in enumerate(class_list):
            
            categorical_label = tf.keras.utils.to_categorical(label_num, num_classes=(len(class_list) + 1))
        
            file_itr = iglob(os.path.join(input_path, label_name, '*.pkl'))
            for filepath in file_itr:

                jobs.append(
                    pool.apply_async(
                        compute_similarities,
                        (filepath, template_streams, threshold, class_list, categorical_label)
                    )
                )
                #i += 1
                #if i > 5:
                #    break
        
        for job in jobs:
            res = job.get()
            preds.append(res[0])
            true_labels.append(res[1])

    true_labels = np.array(true_labels)
    preds = np.array(preds)
    
    true_labels_manyhot = np.argmax(true_labels, axis=1)
    preds_manyhot = np.argmax(preds, axis=1)

    print('accuracy SK:', accuracy_score(true_labels_manyhot, preds_manyhot)) 
    print('precision SK:', precision_score(true_labels_manyhot, preds_manyhot, average='micro')) 
    print('recall SK:', recall_score(true_labels_manyhot, preds_manyhot, average='micro')) 
    
    #tf_recall = tf.keras.metrics.Recall()
    #tf_recall.update_state(true_labels, preds)
    #print('accuracy TF:', tf_recall.result().numpy())

    print('confusion matrix:')
    print(confusion_matrix(true_labels_manyhot, preds_manyhot))

    np.save('true_labels.npy', true_labels)
    np.save('preds.npy', preds)

    print('saved values to', 'true_labels.npy,', 'preds.npy')


def predict(template_streams: Dict[str, Stream], input_path: str, threshold: float):
    """
    Make predictions using all templates

    preds and targets have number of entries equal num classes + 1, where the
    last column represents no detection
    """

    class_list = [d.split('/')[-1] for d in iglob(input_path + '/*')]
    class_list.sort()
    print('classes:', class_list + ['None'])

    true_labels = []
    preds = []

    #i = 0
    for label_num, label_name in enumerate(class_list):

        categorical_label = tf.keras.utils.to_categorical(label_num, num_classes=(len(class_list) + 1))
        
        file_itr = iglob(os.path.join(input_path, label_name, '*.pkl'))
        for filepath in file_itr:

            with open(filepath, 'rb') as fin:
                stream = pickle.load(fin)
            
            stream = preprocess(stream, 3)
            event_preds = []

            for class_name in class_list:
                detections, _ = correlation_detector(
                    stream,
                    template_streams[class_name],
                    heights=threshold,
                    distance=10
                )
                sims = [0]
                for det in detections:
                    sims.append(det['similarity'])
                
                event_preds.append(max(sims))   # Take the best detection
            
            # No detection
            if sum(event_preds) == 0:
                event_preds.append(1)
            else:
                event_preds.append(0)

            
            true_labels.append(categorical_label)
            preds.append(event_preds)
            #print(filepath)
            #print('true:', categorical_label)
            #print('preds:', event_preds)
            
            #i += 1
            #if i > 5:
            #    break

            #print('categorical_label:', categorical_label)
            #print('event_preds:', event_preds)
            ##input('')
    
    true_labels = np.array(true_labels)
    preds = np.array(preds)
    
    true_labels_manyhot = np.argmax(true_labels, axis=1)
    preds_manyhot = np.argmax(preds, axis=1)

    print('accuracy SK:', accuracy_score(true_labels_manyhot, preds_manyhot)) 
    tf_recall = tf.keras.metrics.Recall()
    tf_recall.update_state(true_labels, preds)
    print('accuracy TF:', tf_recall.result().numpy())

    print('confusion matrix:')
    print(confusion_matrix(true_labels_manyhot, preds_manyhot))
    

                






if __name__ == "__main__":




    parser = ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store_true')
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('-e', '--evaluate_preds', action='store_true')
    args = parser.parse_args()

    if args.cluster:
        #cluster_on_reference_station('KIRUNA')
        #cluster_on_reference_station('KEVITSA')
        #cluster_on_reference_station('KIROVSK')
        cluster_on_reference_station('SUURIKUUSIKKO')
        #cluster_on_reference_station('ZAPOLJARNY')
    
    if args.predict:
        

        testing = False
        if testing:

            template_start_kiruna = UTCDateTime('2016-02-10T00:37:27')
            template_start_kevitsa = UTCDateTime('2016-09-27T08:59:58')
            template_start_kirovsk = UTCDateTime('2015-11-13T16:19:48')
            template_start_suurikuusikko = UTCDateTime('2015-11-04T04:01:06') #2015-05-04T13:59:40')
            template_start_zapoljarny = UTCDateTime('2017-02-10T03:56:14')
    
            duration = 60
            duration = 80 # KIROVSK
            duration = 40 # SUURIKUUSIKKO
            duration = 40 # ZAPOLJARNY

            template_start = template_start_suurikuusikko # UTCDateTime('2016-02-10T00:37:27') # UTCDateTime('2015-08-28T23:37:30')
            template_end = template_start + duration

            predict_on_saved_files_OLD(
                '/nobackup/steffen/array-ml-data-may2021/pickle',
                template_start,
                template_end,
                plot=True
            )


        else:
            
            templates = {}
            templates['KEVITSA'] = get_template_stream(UTCDateTime('2016-09-27T08:59:58'), 60)
            templates['KIROVSK'] = get_template_stream(UTCDateTime('2015-11-13T16:19:48'), 80)
            templates['KIRUNA'] = get_template_stream(UTCDateTime('2016-02-10T00:37:27'), 60)
            templates['SUURIKUUSIKKO'] = get_template_stream(UTCDateTime('2015-11-04T04:01:06'), 40)
            templates['ZAPOLJARNY'] = get_template_stream(UTCDateTime('2017-02-10T03:56:14'), 40)

            data_path = '/nobackup/steffen/array-ml-data-may2021/pickle'
            #predict(templates, data_path, threshold=0.2)
            predict_parallel(templates, data_path, threshold=0.05)


    if args.evaluate_preds:
        
        #true_labels = np.load('true_labels_thresh_0p05.npy')
        #preds = np.load('preds_thresh_0p05.npy')
        true_labels = np.load('true_labels_thresh_0p2.npy')
        preds = np.load('preds_thresh_0p2.npy')

        true_labels_manyhot = np.argmax(true_labels, axis=1)
        preds_manyhot = np.argmax(preds, axis=1)

        print('accuracy:', accuracy_score(true_labels_manyhot, preds_manyhot))
        print('precision (micro):', precision_score(true_labels_manyhot, preds_manyhot, average='micro'))
        print('precision (macro):', precision_score(true_labels_manyhot, preds_manyhot, average='macro'))
        print('precision (weighted):', precision_score(true_labels_manyhot, preds_manyhot, average='weighted'))
        print('recall (micro):', recall_score(true_labels_manyhot, preds_manyhot, average='micro'))
        print('recall (macro):', recall_score(true_labels_manyhot, preds_manyhot, average='macro'))
        print('recall (weighted):', recall_score(true_labels_manyhot, preds_manyhot, average='weighted'))

        print('confusion matrix:')
        print(confusion_matrix(true_labels_manyhot, preds_manyhot))










    """
    template_start_time = UTCDateTime('2015-10-13T23:29:32.4')
    template_length = 60

    client = Client()
    #inv = client.get_array_inventory('ARCES', UTCDateTime(2015, 1, 1))
    template = client.get_waveforms(
        'ARA*,ARB*,ARC*,ARD*',
        'BHZ',
        template_start_time,
        template_start_time + template_length
        )
    template.detrend('demean')
    template.taper(0.05)
    template.filter('bandpass', freqmin=3, freqmax=8)

    stream = client.get_waveforms(
        'ARA*,ARB*,ARC*,ARD*',
        'BHZ',
        UTCDateTime('2015-08-28T23:36:48.1') - 120,
        UTCDateTime('2015-08-28T23:36:48.1') + 360,
    )
    stream.detrend('demean')
    stream.taper(0.05)
    stream.filter('bandpass', freqmin=3, freqmax=8)

    
    detections, sims = correlation_detector(
        stream, template,
        heights=0.4, distance=10,
        plot=stream
    )
    """

