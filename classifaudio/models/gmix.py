import glob
import math
import pickle
import random

import numpy as np
import scipy.io.wavfile as wav
from sklearn.mixture import GaussianMixture
from os.path import isdir, join

from classifaudio.features import FeaturesExtractor


class GMix(object):

    def __init__(self, label, components=32, max_iter=200, n_init=2):
        self.label = label
        self.gmm = GaussianMixture(n_components=components, max_iter=max_iter,
                                   n_init=n_init)

    def train_from_directory(self, directory):
        if directory[-1] != "/":
            directory += "/"
        audio_files = glob.glob(directory + "*.wav")
        random.shuffle(audio_files)
        feats = []
        for i in range(len(audio_files)):
            voice = audio_files[i]
            feat = GMix.calculate_mfcc(voice)
            feats.append(feat)
        feats = np.vstack(feats)
        self.gmm.fit(feats)
        return self

    @staticmethod
    def calculate_mfcc(audio_file):
        (rate, sig) = wav.read(audio_file)
        return GMix.extract_features(sig, rate)

    @staticmethod
    def extract_features(audio, rate):
        # not sure?
        # audio, rate = remove_silence(rate, audio)
        return FeaturesExtractor.extract_mfcc_features_from_data(audio, rate,
                                                                 numcep=20)

    def score(self, mfccs):
        if isinstance(mfccs, str):
            mfccs = GMix.calculate_mfcc(mfccs)
        return self.gmm.score(mfccs)

    def save(self, model_path):
        if isdir(model_path):
            model_path = join(model_path, self.label + ".gmix")
        elif not model_path.endswith(".gmix"):
            model_path += ".gmix"
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class GMixInference(object):

    def __init__(self, gmixs):
        self.gmixs = gmixs

    def verify(self, audio_path, claimed_label, threshold=0.5):
        labels = self.gmixs
        try:
            mfccs = GMix.calculate_mfcc(audio_path)
            claimed_label = [x for x in labels if x.label ==
                             claimed_label][0]
            other_labels = [x for x in labels if x.label != claimed_label]
            claimed_label_score = claimed_label.score(mfccs)
            t = [x.score(mfccs) for x in other_labels]
            t = np.array(t)
            other_labels_score = np.exp(t).sum()
            result = math.exp(claimed_label_score) / other_labels_score
            print(result)
            return result >= threshold
        except Exception:
            raise LabelNotFoundException()

    def predict(self, audio_path):
        labels = self.gmixs
        max_score = -1e9
        max_label = None
        mfccs = GMix.calculate_mfcc(audio_path)
        for labl in labels:
            score = labl.score(mfccs)
            if score > max_score:
                max_score = score
                max_label = labl
        return max_label

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class LabelNotFoundException(Exception):
    pass

