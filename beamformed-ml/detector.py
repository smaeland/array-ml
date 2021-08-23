from typing import List, Dict, NamedTuple
import numpy as np
import tensorflow as tf
from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats
from seismonpy.norsardb import Client

detector_config = {
    'station': 'ARCES',
    'channel_order': ['E', 'N', 'Z'],
    'preds_order': ['P', 'S', 'N'],
    'band_and_instr_code': 'BH*',
    'window_length_sec': 5.0,
    'sampling_rate': 40.0,
    'model_file': 'phasenet/model-phasenet-40Hz-jun28',
    'elements': {
        'ARA0': {

        },
        'ARA1': {

        },
        'ARA2': {

        },
    }
}




class DetectorConfig(NamedTuple):
    # TODO convert to dataclass?

    station: str
    channel_order: List[str]
    preds_order: List[str]
    band_and_instr_code: str
    #window_length_sec: float
    sampling_rate: float
    modelfile: str
    elements: List

    @property
    def elements_string(self):
        return ','.join(self.elements.keys())



class ModelWrapper(object):

    def __init__(self, model_path) -> None:
        
        self.model = tf.keras.models.load_model(model_path)
    
    def predict(self, data: np.ndarray):

        return self.model.predict(data)
    
    def get_window_length_pts(self):
        #print(self.model.summary())
        assert self.model.layers[0].input_shape[1] == 6000
        return self.model.layers[0].input_shape[1]




class StationDetector(object):

    def __init__(
        self, 
        station_name,
        station_code,
        channel_order,
        preds_order,
        model: ModelWrapper
    ) -> None:
        """
        Name: full name ('ARCES')
        Code: short identifier ('ARA0')
        """

        self.name = station_name
        self.code = station_code
        self.channel_order = channel_order
        self.preds_order = preds_order
        self.model = model
        self.window_length_pts = self.model.get_window_length_pts()
    

    def detection_stream_from_window(self, processed_stream: Stream) -> Stream:
        """
        Compute detection trace for this stream

        """
        for tr in processed_stream:
            if tr.stats.npts != self.window_length_pts:
                raise RuntimeError(
                    f'Window length is {tr.stats.npts}, expected {self.window_length_pts}'
                )


        out_stream = Stream()

        # Select traces from this station
        st = processed_stream.select(station=self.code)
        
        # Check ordering
        trace_order = [tr.stats.channel[-1] for tr in st.traces]
        if not trace_order == self.channel_order:
            raise RuntimeError(f'Wrong channel order: {trace_order} (expected {self.channel_order})')
            # TODO sort
        
        data = np.transpose(
            np.vstack((st.traces[0], st.traces[1], st.traces[2]))
        ) # -> shape = (npts, 3)
        data = np.expand_dims(data, axis=0) # -> shape = (1, npts, 3)

        preds = self.model.predict(data)[0] # shape = (npts, 3)

        # Construct output traces
        for i in range(preds.shape[1]):
            header = Stats(st.traces[0].stats)
            header.processing.clear()
            header.channel = self.preds_order[i]
            tr = Trace(data=preds[:, i], header=header)
            out_stream.append(tr)
        
        return out_stream
        









class ArrayDetector(object):
    """
    Detect on all array stations using a single CNN
    """

    def __init__(self, config: Dict) -> None:
        
        self.config = DetectorConfig(**config)
        self.dbclient = Client()

        common_model = ModelWrapper(self.config.modelfile)
        self.window_length_pts = common_model.get_window_length_pts()
        self.window_length_sec = self.window_length_pts / self.config.sampling_rate
        self.delta = 1.0/self.config.sampling_rate

        self.detectors = {}
        for elem in self.config.elements.keys():
            self.detectors[elem] = StationDetector(
                elem,
                elem,
                self.config.channel_order,
                self.config.preds_order,
                common_model
            )


    def process_stream(self, stream: Stream, resample_frequency:float, taper_length: float):

        st = Stream(stream)
        st.detrend('demean')
        st.taper(max_percentage=None, max_length=taper_length, type='cosine')
        st.filter('bandpass', freqmin=1.0, freqmax=15.0)
        st.resample(resample_frequency)

        return st


    def load_and_process_stream(self, starttime: UTCDateTime, endtime: UTCDateTime):
        """
        Read in multiples of the window length, then cut to desired length later
        """
        assert endtime > starttime

        duration = endtime - starttime
        start_offset = 0
        if duration % self.window_length_sec == 0:
            end_offset = 0
        else: 
            end_offset = self.window_length_sec - (duration % self.window_length_sec)


        taper_len = 5.0  # sec
        start_offset += taper_len
        end_offset += taper_len

        stream = self.dbclient.get_waveforms(
            self.config.elements_string,
            self.config.band_and_instr_code,
            starttime=starttime - start_offset,
            endtime=endtime + end_offset
        )

        stream = self.process_stream(stream, self.config.sampling_rate, taper_len)

        # Cut the tapered edges
        stream.trim(starttime, endtime + self.window_length_sec)

        # Cut to an optimal length
        surplus_pts = stream.traces[0].stats.npts % self.window_length_pts
        surplus_sec = surplus_pts / self.config.sampling_rate
        endt = stream.traces[0].stats.endtime
        stream.trim(endtime=endt-surplus_sec)

        stream.sort()

        # TODO return desired end time

        return stream




    def detection_stream(self, in_stream: Stream) -> Stream:

        

        # TODO: QC

        out_stream = Stream()


        # Windowing
        window_length_sec = self.window_length_pts / self.config.sampling_rate 
        window_itr = in_stream.slide(
            window_length=window_length_sec - self.delta,
            step=window_length_sec,
            include_partial_windows=True
        )

        for i, window in enumerate(window_itr):
            #print(f'window {i}:')
            #print(window)
            for elem in self.config.elements.keys():
                out_stream += self.detectors[elem].detection_stream_from_window(window)
                
        return out_stream
        

    def create_beam(self, detection_stream: Stream, time_delays: Dict):
        pass



                
if __name__ == '__main__':

    arraydet = ArrayDetector(detector_config)

    start = UTCDateTime('2021-01-07T00:30:36')
    st = Client().get_waveforms('AR*', 'BH*', starttime=start, endtime=start+180)

    detstream = arraydet.detection_stream(st, 1.0)

    print(detstream)
    detstream.plot()
