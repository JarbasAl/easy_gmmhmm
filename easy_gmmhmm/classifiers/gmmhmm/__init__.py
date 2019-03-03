from easy_gmmhmm.trainer import Trainer
from easy_gmmhmm.tester import Tester
from easy_gmmhmm.utils import folder2mfcc
from hmmlearn.hmm import GMMHMM
import numpy as np
from os.path import join


class GMMHMMTrainer(Trainer):
    """
   Class that takes in a directory containing training data as raw wavfiles
   within folders named according to label and extracts MFCC feature vectors from them,

   Accepts a configuration for each in terms of number of states for HMM
   and number of mixtures in the Gaussian Model and then trains a set of
   GMMHMMs, one for each label.

   Creates and save as pickle A python dictionary of GMMHMMs that are
   trained, key values being labels extracted from folder names.
   """
    def __init__(self, data_path, model_name="gmmhmm", models_path="models",
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
        print(label)
        # TODO suggest good defaults
        conf["n_components"] = int(eval(input(
            "Enter number of components in the GMMHMM: ")))
        conf["n_mix"] = int(eval(input(
            "Enter number of mixtures in the Gaussian Model: ")))
        return conf

    def get_model(self, config):
        """
        Initialize a GMMHMM model from config

        Args:
            config: dictionary of params to build the model

        Returns:
            model (GMMHMM)

        """
        # create GMM object from config
        return GMMHMM(**config)

    def train_model(self, model_obj, trng_data):
        """

        Args:
            model_obj (GMMHMM): model to be trained
            trng_data: features to train the model with

        Returns:
            trained model (GMMHMM)
        """
        model_obj.fit(np.vstack(trng_data))
        return model_obj

    def get_training_data_features(self, pickle_path=None):
        """
        Generates features from data folder

        if pickle path is provided those are pickled and saved, useful to
        retrain with different params

        Args:
            pickle_path: path to save pickled data, defaults to model_path/model_name_trng_data.pkl

        Returns:
            mfcc_features (dict): dict of label : extracted features list
        """
        save = pickle_path is not None
        pickle_path = pickle_path or join(self.models_path, self.model_name +
                                          '_trng_data.pkl')
        print("preparing training data features")
        return folder2mfcc(self.labels, self.data_path,
                           pickle_path=pickle_path,
                           write_pickle=save,
                           extract_config=self.extract_config)


class GMMHMMTester(Tester):
    """
    Load a GMMHMM model and predict labels
    """

    def test_file(self, test_file, gmms):
        """
        Load a given file and predict a label for it.

        Args:
            test_file (str): path to wav file to test
            gmms (dict): models to use for inference
        """
        mfcc_feat = self.read_features(test_file)
        pred = {}
        for model in gmms:
            pred[model] = gmms[model].score(mfcc_feat)
        return self.get_nbest(pred, 2), pred


