# LATSS

A subject-independent motor imagery classification model utilizing advanced domain adaptation techniques.

## Description

LATSS is a subject-independent motor imagery classification model that utilizes advanced domain adaptation techniques to improve the generalization of the model across subjects. The model employs latent alignment to align source domain data to a target domain with the help of some labeled 'calibration' data. Tangent space mapping is then used as a feature extraction technique. The extracted features are then fed into an SVM classifier with an RBF kernel to classify the motor imagery data.

## Installation

```bash
$ pip install latss
```

## Usage

Training and predicting with the LATSS model is simple. Here's an example of how to use it:

```python
from latss import LATSS

# Load source data
source_data = ...

# Initialize the model
model = LATSS(source_data=source_data)

# Calibrate and train the model
# Note: calibration_data must be an annotated mne.io.Raw object
calibration_data = ...
model.calibrate(calibration_data)

# Predict on new data
# Note: new_data must be a mne.io.Raw object as well
new_data = ...
prediction = model.predict(new_data)
```

  
Source data must either be an mne.Epochs object or a dictionary of {data: np.ndarray, labels: np.ndarray} pairs.
Both of input options must match model requirements.    
To load source datasets, you can either:

1. Use the `load_data()` function to download "-epo.fif" files by passing
a dictionary of {filename-epo.fif: url} pairs. 

```python
from latss import load_data

data = {
    'filename-epo.fif': 'url',
    'filename-2-epo.fif': 'url'
}

source_data = load_data(data)
```

2. Load your own "-epo.fif" files using the `unpack_fifs()` function.

```python
from latss import unpack_fifs

fnames = ['path/to/filename-epo.fif',
          'path/to/filename-2-epo.fif']

source_data = unpack_fifs(fnames)
```

3. Or you can optionally use any mne.Epochs object as your source dataset as desired.

## License

`latss` was created by Zeyad Ahmed. It is licensed under the terms
of the MIT license.

## Credits

The LATSS model was inspired by the work of He et al. [1], while introducing some key modifications and improvements.

[1] H. He and D. Wu, "Different Set Domain Adaptation for Brain-Computer Interfaces: A Label Alignment Approach," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 28, no. 5, pp. 1091-1108, May 2020, doi: [10.1109/TNSRE.2020.2980299](https://doi.org/10.1109/TNSRE.2020.2980299).

