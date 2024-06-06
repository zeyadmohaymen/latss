from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mne import Epochs, EpochsArray
from mne.io import Raw, BaseRaw, RawArray
from latss.preprocessing.raw_preprocessing import FilterRaw, RemoveArtifacts, Epochify
from latss.preprocessing.epochs_preprocessing import Resampler, IntraEpochSegmentation
from latss.label_alignment.la import LabelAlignment
from latss.tsmapping.tsm import TangentSpaceMapping
from latss.utils.utils import validate_raw, validate_epochs, validate_dict, validate_input_type

class LATSS:
    """
    Label Alignment - Tangent Space (Mapping) - Support Vector Machine classifier, LATSS for short.

    Parameters:
    - source_data (mne.Epochs | dict): Source data to be used for label alignment.
    - sfreq (int): Sampling frequency of the data (default: 160).
    - epoch_length (int): Length of each epoch in seconds (default: 2).
    - window_size (int): Size of the sliding window in seconds (default: 1).
    - window_overlap (int): Overlap between co2nsecutive windows in seconds (default: 0.2).
    - svm_C (float): Regularization parameter for the SVM classifier (default: 100).

    Methods:
    - get_params(): Returns the parameters of the classifier.
    - calibrate(raw): Calibrates the classifier using the provided raw data.
    - predict(raw): Predicts the labels for the provided raw data.

    Example:
    ```python
    from latss import LATSS

    # Load source data
    source_data = ...

    # Initialize the model
    model = LATSS(source_data=source_data)

    # Calibrate and train the model
    calibration_data = ...
    event_id = {
                'left_hand': 1,
                'right_hand': 2,
                }
    acc = model.calibrate(calibration_data, event_id=event_id)

    # Predict on new data
    new_data = ...
    prediction = model.predict(new_data)
    ```
    """

    def __init__(self, source_data: Epochs | dict, sfreq=160, epoch_length=2, window_size=1, window_overlap=0.2, svm_C=100):
        self._sfreq = sfreq
        self._epoch_length = epoch_length
        self._window_size = window_size
        self._window_overlap = window_overlap

        self._source_data, self._source_events = self._handle_input(source_data)

        self._clf = Pipeline([
            ('tsm', TangentSpaceMapping()),
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=svm_C))
        ])

    def get_params(self):
        """
        Returns the parameters of the classifier.

        Returns:
        - dict: Dictionary containing the parameters of the classifier.
        """
        return {
            'sfreq': self._sfreq,
            'epoch_length': self._epoch_length,
            'window_size': self._window_size,
            'window_overlap': self._window_overlap
        }
    
    def calibrate(self, raw: Raw, event_id: dict = {'rest': 0, 'feet': 1}):
        """
        Calibrates the classifier using the provided raw data.
        - Raw data is preprocessed
        - Data is split into calibration and test sets
        - Calibration set is used to align the source data using label alignment
        - Classifier is trained on the aligned data
        - Accuracy score of the classifier is returned

        Parameters:
        - raw: MNE Raw object to be used for calibration.
        - event_id: Dictionary containing the annotations and their desired corresponding event IDs. {annotation: event_id (int)}

        Returns:
        - float: Accuracy score of the classifier.
        """
        validate_input_type(raw, [Raw, BaseRaw, RawArray])

        # Preprocess the raw data
        calib_data, calib_labels = self._preprocess(raw, event_id = event_id)

        # Split the data into calibration and test sets
        calib_data, test_data, calib_labels, test_labels = train_test_split(calib_data, calib_labels, test_size=0.2, random_state=42)

        # Align the source data using label alignment
        la = LabelAlignment(calib_data, calib_labels)
        aligned_data, aligned_labels = la.fit_transform(self._source_data, self._source_events)

        # Train the classifier
        self._clf.fit(aligned_data, aligned_labels)

        return self._clf.score(test_data, test_labels)
    
    def predict(self, raw: Raw):
        """
        Predicts the labels for the provided raw data

        Parameters:
        - raw: MNE Raw object for which labels are to be predicted.

        Returns:
        - array: Predicted labels for the raw data.
        """
        validate_input_type(raw, [Raw, BaseRaw, RawArray])
        
        preprocessed_data, _ = self._preprocess(raw, epochify=False)

        return self._clf.predict(preprocessed_data)

    def _preprocess(self, raw, epochify=True, event_id=None):
        """
        Preprocesses the raw data.

        Parameters:
        - raw: Raw data to be preprocessed.

        Returns:
        - tuple: Tuple containing the preprocessed epoch data and labels.
        """

        pipe = Pipeline([
            ('remove_artifacts', RemoveArtifacts(n_components=5)),
            ('filter', FilterRaw()),
            ('epochify', Epochify(event_id=event_id, length=self._epoch_length) if epochify else 'passthrough'),
            ('resample', Resampler(self._sfreq) if raw.info['sfreq'] != self._sfreq else 'passthrough'), #! Will incoming Raw object have sfreq attribute?
            ('segmenter', IntraEpochSegmentation(self._window_size, self._window_overlap) if self._window_size and self._window_overlap else 'passthrough'),
        ])

        epoch_data, event_data = pipe.fit_transform(raw)
        labels = event_data[:, -1] if event_data.ndim == 2 else event_data

        return epoch_data, labels

    
    def _handle_input(self, source):
        """
        Handles the input source data and validates it.

        Parameters:
        - source: Source data to be validated.

        Returns:
        - tuple: Tuple containing the source data and events.

        Raises:
        - ValueError: If the source data type is invalid.
        """
        data = None
        events = None
        if isinstance(source, Epochs) or isinstance(source, EpochsArray):
            validate_epochs(source, self._sfreq, self._epoch_length, self._window_size)
            data = source.get_data(copy=False)
            events = source.events[:, -1]
        elif isinstance(source, dict):
            validate_dict(source, self._sfreq, self._epoch_length, self._window_size)
            data = source['data']
            events = source['events'][:, -1]
        else:
            raise ValueError("Invalid source data type: Expected mne.Epochs or dict")
        
        return data, events