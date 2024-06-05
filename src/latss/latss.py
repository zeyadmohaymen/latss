from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mne
from latss.preprocessing.raw_preprocessing import FilterRaw, RemoveArtifacts, Epochify
from latss.preprocessing.epochs_preprocessing import Resampler, IntraEpochSegmentation, EpochsDecoder
from latss.label_alignment.la import LabelAlignment
from latss.tsmapping.tsm import TangentSpaceMapping
from latss.utils.utils import validate_raw, validate_epochs

class LATSS:
    """
    Label Alignment - Tangent Space (Mapping) - Support Vector Machine classifier, LATSS for short.

    Parameters:
    - source_data (dict): Dictionary containing the source data and events.
    - sfreq (int): Sampling frequency of the data (default: 160).
    - epoch_length (int): Length of each epoch in seconds (default: 2).
    - window_size (int): Size of the sliding window in seconds (default: 1).
    - window_overlap (int): Overlap between co2nsecutive windows in seconds (default: 0.2).
    - svm_C (float): Regularization parameter for the SVM classifier (default: 100).

    Methods:
    - get_params(): Returns the parameters of the classifier.
    - calibrate(raw): Calibrates the classifier using the provided raw data.
    - predict(raw): Predicts the labels for the provided raw data.

    Private Methods:
    - _validate(raw): Validates the raw data.
    - _preprocess(raw): Preprocesses the raw data.
    - _preprocess_raw(raw): Preprocesses the raw data without label alignment.
    """

    def __init__(self, source_epochs: mne.Epochs, sfreq=160, epoch_length=2, window_size=1, window_overlap=0.2, svm_C=100):
        validate_epochs(source_epochs, sfreq, epoch_length, window_size)

        self._source_data = source_epochs.get_data(copy=False)
        self._source_events = source_epochs.events[:, -1]
        self._sfreq = sfreq
        self._epoch_length = epoch_length
        self._window_size = window_size
        self._window_overlap = window_overlap

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
    
    def calibrate(self, raw: mne.io.Raw):
        """
        Calibrates the classifier using the provided raw data.
        - Raw data is preprocessed
        - Data is split into calibration and test sets
        - Calibration set is used to align the source data using label alignment
        - Classifier is trained on the aligned data
        - Accuracy score of the classifier is returned

        Parameters:
        - raw: Raw data to be used for calibration.

        Returns:
        - float: Accuracy score of the classifier.
        """
        # Preprocess the raw data
        calib_data, calib_labels = self._preprocess(raw)

        # Split the data into calibration and test sets
        calib_data, test_data, calib_labels, test_labels = train_test_split(calib_data, calib_labels, test_size=0.2, random_state=42)

        # Align the source data using label alignment
        la = LabelAlignment(calib_data, calib_labels)
        aligned_data, aligned_labels = la.fit_transform(self._source_data, self._source_events)

        # Train the classifier
        self._clf.fit(aligned_data, aligned_labels)

        return self._clf.score(test_data, test_labels)
    
    def predict(self, raw: mne.io.Raw):
        """
        Predicts the labels for the provided raw data. Data must conform to model's parameters.

        Parameters:
        - raw: Raw data for which labels are to be predicted.

        Returns:
        - array: Predicted labels for the raw data.

        Raises:
        - ValueError: If the epoch length of the raw data is invalid.
        """
        validate_raw(preprocessed_raw, self._sfreq, self._epoch_length, self._window_size)

        preprocessed_raw = self._preprocess_raw(raw)
        preprocessed_data = preprocessed_raw.get_data()

        return self._clf.predict(preprocessed_data) #! Should return ONE label

    def _preprocess(self, raw):
        """
        Preprocesses the raw data.

        Parameters:
        - raw: Raw data to be preprocessed.

        Returns:
        - tuple: Tuple containing the preprocessed epoch data and labels.
        """
        initial_preprocessing = self._preprocess_raw(raw)

        pipe = Pipeline([
            ('epochify', Epochify(length=self._epoch_length)),
            ('resample', Resampler(self._sfreq)) if raw.info['sfreq'] != self._sfreq else None, #! Will incoming Raw object have sfreq attribute?
            ('segmenter', IntraEpochSegmentation(self._window_size, self._window_overlap)) if self._window_size and self._window_overlap else None,
            ('decoder', EpochsDecoder()),            
        ])

        epoch_data, event_data = pipe.fit_transform(initial_preprocessing)
        labels = event_data[:, -1]

        return epoch_data, labels
    
    def _preprocess_raw(self, raw):
        """
        Initial preprocessing of the raw data. This includes filtering and removing artifacts.

        Parameters:
        - raw: Raw data to be preprocessed.

        Returns:
        - Preprocessed raw data.
        """
        pipe = Pipeline([
            ('remove_artifacts', RemoveArtifacts(n_components=5)),
            ('filter', FilterRaw()),
        ])

        return pipe.fit_transform(raw)