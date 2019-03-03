import scipy.io.wavfile as wvf
import glob
import os
import pickle
from easy_gmmhmm.features import FeaturesExtractor


def read_wav(wavfile):
    """
    Utility wrapper around scipy.io.wavfile.read

    Args:
        wavfile (str): wav file to load

    Returns:
        wav_data
    """
    return wvf.read(wavfile)


def folder2mfcc(labels, data_path, pickle_path="trng_data.pkl",
                from_file=False, write_pickle=False, extract_config=None):
    """
    Utility function to read wav files, convert them into MFCC vectors and store in a pickle file
    (Pickle file is useful in case you re-train on the same data changing hyperparameters)

    Args:
        labels (list): list of labels (folder names) to look for in data_path
        data_path (str): path to load features from
        pickle_path (str): path to save/load pickled features
        from_file (bool): try to load pickle instead of extracting
        write_pickle (bool): save pickled features
        extract_config (dict): config for extracting features

    Returns:
        mfcc_features (dict): dict of label: [extracted_features]
    """
    trng_data = {}
    if from_file and os.path.isfile(pickle_path):
        write_pickle = False
        trng_data = pickle.load(open(pickle_path, "rb"))
    else:
        for lbl in labels:
            mfccs = []
            for wavfile in glob.glob(data_path + '/' + lbl + '/*.wav'):
                if extract_config:
                    mfcc_feat = FeaturesExtractor.extract_mfcc_features(
                        wavfile, **extract_config)
                else:
                    mfcc_feat = FeaturesExtractor.extract(wavfile)
                mfccs.append(mfcc_feat)
            trng_data[lbl] = mfccs
    if write_pickle:
        pickle.dump(trng_data, open(pickle_path, "wb"))
    return trng_data
