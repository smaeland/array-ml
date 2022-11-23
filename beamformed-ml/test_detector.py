import pytest
from obspy import UTCDateTime
from seismonpy.norsardb import Client
from detector import (
    DetectorConfig,
    ModelWrapper,
    StationDetector,
    ArrayDetector
)


@pytest.fixture
def config_dict():
    
    c = {
        'station': 'ARCES',
        'channel_order': ['E', 'N', 'Z'],
        'preds_order': ['P', 'S', 'N'],
        'band_and_instr_code': 'BH*',
        #'window_length_sec': 5.0,
        'sampling_rate': 100.0,
        'modelfile': 'phasenet/testmodel-phasenet',
        'elements': {
            'ARA0': {

            },
            'ARA1': {

            },
            'ARA2': {

            },
        }
    }

    return c


@pytest.fixture
def test_data_kiruna_short():

    # TODO: take params from config
    arrival_time_arces = UTCDateTime('2020-05-18T01:12:39.7')
    st = Client().get_waveforms(
        'ARA*', 'HH*', arrival_time_arces-5, arrival_time_arces+55
    )
    st.detrend('demean')
    st.resample(100.0)

    # Remove last sample, to make length 6000 pts
    delta = st.traces[0].stats.delta
    endt = st.traces[0].stats.endtime
    st.trim(endtime=endt-delta)

    return st



@pytest.fixture
def test_data_kiruna_long():

    # TODO: take params from config
    arrival_time_arces = UTCDateTime('2020-05-18T01:12:39.7')
    st = Client().get_waveforms(
        'ARA*', 'HH*', arrival_time_arces-60, arrival_time_arces+180
    )
    st.detrend('demean')
    st.resample(100.0)
    
    # Remove last sample
    delta = st.traces[0].stats.delta
    endt = st.traces[0].stats.endtime
    st.trim(endtime=endt-delta)

    return st


@pytest.fixture
def test_data_kiruna_short():

    # TODO: take params from config
    arrival_time_arces = UTCDateTime('2020-05-18T01:12:39.7')
    st = Client().get_waveforms(
        'ARA*', 'HH*', arrival_time_arces-5, arrival_time_arces+55
    )
    st.detrend('demean')
    st.resample(100.0)

    # Remove last sample to make length exactly 6000 pts
    delta = st.traces[0].stats.delta
    endt = st.traces[0].stats.endtime
    st.trim(endtime=endt-delta)

    return st



class TestDetectorConfig:

    def test_elements_string(self, config_dict):

        dc = DetectorConfig(**config_dict)

        assert dc.elements_string == 'ARA0,ARA1,ARA2'



class TestStationDetector:

    def test_detection_stream(self, test_data_kiruna_short):
        # TODO: take params from config


        model = ModelWrapper('phasenet/testmodel-phasenet')
        sd = StationDetector(
            'ARCES', 'ARA0', ['E', 'N', 'Z'], ['P', 'S', 'N'], model
        )
        det_stream = sd.detection_stream(test_data_kiruna_short)
        print(det_stream)
        det_stream.plot()


class TestArrayDetector:

    def test_init(self, config_dict):

        ad = ArrayDetector(config_dict)

        assert ad.config.station == 'ARCES'

    
    def test_load_and_process_stream(self, config_dict):
        """
        load_and_process_stream should read one window too far, so
        with a window length of 60 s we should get a 120 s long stream
        """

        ad = ArrayDetector(config_dict)
        start = UTCDateTime('2020-05-18T01:12:39.7')
        end = start + 67
        stream = ad.load_and_process_stream(start, end)

        expected_npts = ad.window_length_pts * 2
        expected_npts = int(expected_npts)

        assert stream.traces[0].stats.npts == expected_npts


    def test_detection_stream(self, config_dict, test_data_kiruna_long):

        ad = ArrayDetector(config_dict)
        det_stream = ad.detection_stream(test_data_kiruna_long)
        print(det_stream)
        det_stream.plot()
    


