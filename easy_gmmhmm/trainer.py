import pickle
import os
from os.path import join


class Trainer(object):
    """
    Base interface to load/train models from folders with .wav files

    Handles pickling features/config/models
    """

    def __init__(self, data_path, model_name, models_path="models",
                 config=None, extract_config=None):
        """

        Args:
            data_path (str): Path to the training wav files. Each folder in this path is a label and must NOT be empty.
            model_name (str) : Name of this model, will be saved as model_name.pkl
            models_path (str): Path to store the generated pickle files in.
            config (dict) : Params to initialize models, if not provided will be asked interactively
            extract_config (dict) : Params to extract features

        """
        self.data_path = data_path
        self.model_name = model_name
        self.models_path = models_path
        self.config = config
        self.extract_config = extract_config

    def ask_config(self, label):
        """
        Get config parameters interactively

        Args:
            label (str) : label we are asking params for

        Returns (dict) : config to build the model

        """
        raise NotImplementedError

    def obtain_config(self, file_path=None, from_file=False, save=False):
        """
         Utility function to take in parameters to train individual models

        Args:
            file_path (str): path to save/load config pickle, default model_path/model_name_conf.pkl
            from_file (bool): load from file instead of asking user
            save (bool): save pickled config

        Returns:

        """
        file_path = file_path or join(self.models_path, self.model_name +
                                      '_conf.pkl')
        conf = {}
        if not from_file:
            if self.config is None:
                for label in self.labels:
                    conf[label] = self.ask_config(label)
            else:
                conf = self.config
            if save:
                pickle.dump(conf, open(file_path, "wb"))
        else:
            conf = pickle.load(open(file_path, "rb"))
        return conf

    def get_model(self, config):
        """
        Initialize model from config

        Args:
            config: dictionary of params to build the model

        Returns:
            model

        """
        raise NotImplementedError

    def train_model(self, model_obj, trng_data):
        """
       Args:
           model_obj: model to be trained
           trng_data: features to train the model with

       Returns:
           trained model
       """
        # TODO return trained model
        raise NotImplementedError

    def prepare_models(self, features_dict=None, model_config=None,
                       model_path=None, from_file=False, save=True):
        """
        Utility function to train or load models based on entered configuration and training data.

        Args:
            features_dict (dict): dict of label: features to train from
            model_config: (dict): params of models label: config_dict
            model_path (str): path of the model to save/load from
            from_file (bool): load from file instead of training
            save (bool): save pickled model when trained

        Returns:
            models: dictionary of label: trained object
        """
        
        model_path = model_path or join(self.models_path,
                                        self.model_name + '.pkl')
        models = {}
        if not from_file:
            for label in self.labels:
                model = self.get_model(model_config[label])
                print("training", self.model_name, "for label:", label)
                models[label] = self.train_model(model, features_dict[label])
            if save:
                pickle.dump(models, open(model_path, "wb"))
        else:
            print("loading model from file", model_path)
            models = pickle.load(open(model_path, "rb"))
        return models

    def get_training_data_features(self, pickle_path=None):
        """
       Generates features from data folder

       if pickle path is provided those are pickled and saved, useful to
       retrain with different params

       Args:
           pickle_path: path to save pickled data, defaults to model_path/model_name_trng_data.pkl

       Returns:
           features (dict): dict of label : extracted features list
       """
        # TODO load features
        raise NotImplementedError

    @property
    def labels(self):
        """
        Lookup labels from training data path

        Returns:
            labels (list)
        """
        return os.listdir(self.data_path)

    def train(self, save=True):
        """

        prepare features, obtain config, train models

        Returns:
            models (dict): dictionary of label: trained model
        """
        trng_pickle_path = None
        if save:
            trng_pickle_path = join(self.models_path, self.model_name +
                                    '_trng_data.pkl')
        trng_data = self.get_training_data_features(trng_pickle_path)

        conf_pickle_path = join(self.models_path, self.model_name +
                                '_conf.pkl')
        model_config = self.obtain_config(file_path=conf_pickle_path,
                                          save=save)

        model_path = join(self.models_path, self.model_name + '.pkl')
        models = self.prepare_models(trng_data, model_config,
                                     model_path=model_path, save=save)
        return models

    def load_or_train(self):
        """

        load or prepare features, load or obtain config, load or train models

        Returns:
           models (dict): dictionary of label: trained model
        """
        trng_pickle_path = join(self.models_path, self.model_name +
                                '_trng_data.pkl')
        trng_data = self.get_training_data_features(trng_pickle_path)

        conf_pickle_path = join(self.models_path, self.model_name +
                                '_conf.pkl')
        model_config = self.obtain_config(file_path=conf_pickle_path,
                                          from_file=True)

        model_path = join(self.models_path, self.model_name + '.pkl')
        models = self.prepare_models(trng_data, model_config,
                                     model_path=model_path, from_file=True)
        return models
