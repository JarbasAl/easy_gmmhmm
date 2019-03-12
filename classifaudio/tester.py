import pickle
import heapq
from classifaudio.features import FeaturesExtractor


class Tester(object):
    """
    Base class to get predictions from a model

    """

    def __init__(self, model_path, extract_config=None):
        """

        Args:
            model_path (str): Model to use for inference
        """
        self.model_path = model_path
        self.models = self.load()
        self.extract_config = extract_config or {}

    def load(self, model_path=None):
        """
        Load a model from file

        Args:
          model_path (str): pickled model path

        Returns:
             loaded_models (dict): label: model
        """
        model_path = model_path or self.model_path
        return pickle.load(open(model_path, "rb"))

    def read_features(self, test_file):
        """

        Args:
            test_file (str): wav file to extract features from

        Returns:
            features: extracted mfcc features for test_file
        """
        return FeaturesExtractor.extract_mfcc_features(test_file,
                                                       **self.extract_config)

    def test_file(self, test_file, models):
        """
        Load a given file and predict a label for it.

        Args:
            test_file (str): path to wav file to test
            models (dict): models to use for inference
        """
        raise NotImplementedError

    def predict_label(self, test_file):
        """
        predict label for input wav file.

        Args:
            test_file (str): path to Wav file for which label should be predicted.

        Return:
            A list of predicted label and next best predicted label.
        """
        predicted = self.test_file(test_file, self.models)
        return predicted

    @staticmethod
    def get_nbest(d, n):
        """
        Utility function to return n best predictions.

        Args:
            d (dict): predictions
            n (int): number of predictions to return

        Returns:
            n best predictions
        """
        return heapq.nlargest(n, d, key=lambda k: d[k])
