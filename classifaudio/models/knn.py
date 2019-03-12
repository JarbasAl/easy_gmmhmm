import glob
import pickle
import numpy as np
import scipy.io.wavfile as wav
from sklearn.neighbors import KNeighborsClassifier
from os.path import isdir, join

from classifaudio.features import FeaturesExtractor


class KNC(object):

    def __init__(self, name, labels, **kwargs):
        self.labels = labels
        self.name = name
        self.knn = KNeighborsClassifier(**kwargs)

    def train_from_directory(self, directory):
        X = []
        y = []
        for label in self.labels:

            for fn in glob.glob(join(directory, label, "*.wav")):
                ceps = KNC.calculate_mfcc(fn)
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
                y.append(label)
        self.knn.fit(np.array(X), np.array(y))
        return self

    @staticmethod
    def calculate_mfcc(audio_file):
        (rate, sig) = wav.read(audio_file)
        return KNC.extract_features(sig, rate)

    @staticmethod
    def extract_features(audio, rate):
        # not sure?
        # audio, rate = remove_silence(rate, audio)
        return FeaturesExtractor.extract_mfcc_features_from_data(audio, rate,
                                                                 numcep=20)

    def predict(self, mfccs):
        if isinstance(mfccs, str):
            mfccs = KNC.calculate_mfcc(mfccs)
        return self.knn.predict(mfccs)[0]

    def save(self, model_path):
        if isdir(model_path):
            model_path = join(model_path, self.name + ".knn")
        elif not model_path.endswith(".knn"):
            model_path += ".knn"
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class KNCInference(KNC):

    def verify(self, audio_path, claimed_label, threshold=0.5):
        if claimed_label not in self.labels:
            raise LabelNotFoundException
        if self.predict(audio_path) == claimed_label:
            return True
        return False

    def predict(self, audio_path):
        mfccs = KNC.calculate_mfcc(audio_path)
        return KNC.predict(self, mfccs)


class LabelNotFoundException(Exception):
    pass
