# LATSS

A subject-independent motor imagery classification model.

## Description

Label Alignment - Tangent Space Mapping - SVM, or LATSS for short, is a subject-independent motor imagery classification model that utilizes advanced domain adaptation techniques to improve the generalization of the model across subjects.

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
event_id = {
            'left_hand': 1,
            'right_hand': 2,
            }
acc = model.fit(calibration_data, event_id=event_id)

# Predict on new data
# Note: new_data must be a mne.io.Raw object as well
new_data = ...
prediction = model.predict(new_data)
```
  
  
Source data can be any [mne.Epochs](https://mne.tools/stable/generated/mne.Epochs.html) object or a dictionary with the following structure:
```python
{
    'data': np.array,  # shape: (n_trials, n_channels, n_samples)
    'labels': np.array,  # shape: (n_events, 3)
}
```


## License

`latss` was created by Zeyad Ahmed. It is licensed under the terms
of the MIT license.

## Credits

The LATSS model was inspired by the work of He et al. [1], while introducing some key modifications and improvements.

[1] H. He and D. Wu, "Different Set Domain Adaptation for Brain-Computer Interfaces: A Label Alignment Approach," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 28, no. 5, pp. 1091-1108, May 2020, doi: [10.1109/TNSRE.2020.2980299](https://doi.org/10.1109/TNSRE.2020.2980299).

