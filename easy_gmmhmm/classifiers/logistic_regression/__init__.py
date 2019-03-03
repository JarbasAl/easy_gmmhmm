from sklearn import linear_model
import os
import glob
from easy_gmmhmm.trainer import Trainer
from easy_gmmhmm.tester import Tester
from easy_gmmhmm.features import FeaturesExtractor
import pickle
import numpy as np
from os.path import join


class LogisticRegressionTrainer(Trainer):
    """
   Class that takes in a directory containing training data as raw wavfiles
   within folders named according to label and extracts MFCC feature vectors from them,

   Creates and save as pickle
   """
    def __init__(self, data_path, model_name="logreg", models_path="models",
                 config=None, extract_config=None):
        """

        Args:
            data_path (str): Path to the training wav files. Each folder in this path is a label and must NOT be empty.
            model_name (str) : Name of this model, will be saved as model_name.pkl
            models_path (str): Path to store the generated pickle files in.
            config (dict) : Params to initialize models, if not provided will be asked interactively
            extract_config (dict) : Params to extract features
        """
        Trainer.__init__(self, data_path, model_name, models_path,
                         config, extract_config)

    def ask_config(self, label=""):
        """
        Get config parameters interactively

        Args:
            label (str) : label we are asking params for

        Returns (dict) : config to build the model

        """
        conf = {}
        return conf

    def get_model(self, config):
        """
        Initialize a LogisticRegression model from config

        Args:
            config: dictionary of params to build the model

        Returns:
            model (LogisticRegression)

        """
        # create LogisticRegression object from config
        return linear_model.logistic.LogisticRegression(**config)

    def train_model(self, model_obj, trng_data):
        """

        Args:
            model_obj (LogisticRegression): model to be trained
            trng_data: features to train the model with

        Returns:
            trained model (LogisticRegression)
        """
        model_obj.fit(trng_data, np.array(self.labels))
        return model_obj

    def get_training_data_features(self, pickle_path=None):
        """
        Generates features from data folder

        if pickle path is provided those are pickled and saved, useful to
        retrain with different params

        Args:
            pickle_path: path to save pickled data, defaults to model_path/model_name_trng_data.pkl

        Returns:
            features (numpy.array)
        """
        save = pickle_path is not None
        pickle_path = pickle_path or join(self.models_path, self.model_name +
                                          '_trng_data.pkl')
        print("preparing training data features")
        X = []
        for label in self.labels:
            for fn in glob.glob(os.path.join(self.data_path, label,
                                             "*.wav")):
                ceps = FeaturesExtractor.extract(fn)
                # ceps, mspec, spec= mfcc(song_array)
                # this is done in order to replace NaN and infinite value in array
                bad_indices = np.where(np.isnan(ceps))
                b = np.where(np.isinf(ceps))
                ceps[bad_indices] = 0
                ceps[b] = 0

                num_ceps = len(ceps)
                X.append(np.mean(
                    ceps[int(num_ceps * 1 / 10):int(num_ceps * 9 / 10)],
                    axis=0))
        features = np.array(X)
        if save:
            pickle.dump(features, open(pickle_path, "wb"))
        return features


class LogisticRegressionTester(Tester):
    """
    Load a LogisticRegression model and predict labels
    """

    def test_file(self, test_file, logreg):
        """
        Load a given file and predict a label for it.

        Args:
            test_file (str): path to wav file to test
            knns (dict): models to use for inference
        """
        mfcc_feat = self.read_features(test_file)
        pred = logreg.predict(mfcc_feat)
        return pred


