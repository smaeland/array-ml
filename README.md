# Array-ML

### Setup 
```
pip install -r requirements.txt 
```

### Beamforming of ML output 

Do beamforming of ML-based detection traces for stations in the ARCES array,
similar to what was done for correlation traces in 
https://doi.org/10.1111/j.1365-246X.2006.02865.x

Use a CNN model based on [PhaseNet](https://doi.org/10.1093/gji/ggy423),
trained on [STEAD](https://doi.org/10.1109/ACCESS.2019.2947848) data.
The STEAD data doesn't match our sampling rate, so we downsample to
40 Hz before training a model, using `beamformed-ml/phasenet/resample_stead_files.py`.
Training is done in `beamformed-ml/phasenet/phasenet.py`,
but one can also just use the one provided here 
(`beamformed-ml/phasenet/model-phasenet-40Hz-jun28`), as long as the same
TensorFlow version is used (2.3.1).

To run the code on an example event in Kiruna, using the CNN model
provided in this repo, run
```
cd beamformed-ml
python detector.py
```

### 2D CNNs for seismic arrays

The code in `nn-2d` was used for the SnT2021 presentation 
"P3.6-326 – A neural network architecture for detecting repeating events using
seismic arrays" and is again (pre)trained on STEAD data. Preparing the 
STEAD data is done in `nn-2d/convert_stead_files_to_single_chunk.py`, and
pre-training is done by running 
```
cd nn-2d
python train.py --pretrain_on_stead_data
```
To run on ARCES data, one can take inspiration from `extract_data.py` for how to
use the [Helsinki bulletin](https://www.seismo.helsinki.fi/bulletin/list/norBull.html)
to find relevant events, and save them as pickle files. The `train.py` file can
then be run to do further operations on these data, including training the final model.
It is probably best, however, to just email me and I can give additional instructions.

