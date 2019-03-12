import scipy.io.wavfile as wvf
import glob
import os
import pickle
import numpy as np
from classifaudio.features import FeaturesExtractor


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


def remove_silence(fs,
                   signal,
                   frame_duration=0.02,
                   frame_shift=0.01,
                   perc=0.15):
    orig_dtype = type(signal[0])
    typeinfo = np.iinfo(orig_dtype)
    is_unsigned = typeinfo.min >= 0
    signal = signal.astype(np.int64)
    if is_unsigned:
        signal = signal - (typeinfo.max + 1) / 2
    siglen = len(signal)
    retsig = np.zeros(siglen, dtype=np.int64)
    frame_length = int(frame_duration * fs)
    frame_shift_length = int(frame_shift * fs)
    new_siglen = 0
    i = 0
    average_energy = np.sum(signal ** 2) / float(siglen)
    while i < siglen:
        subsig = signal[i:i + frame_length]
        ave_energy = np.sum(subsig ** 2) / float(len(subsig))
        if ave_energy < average_energy * perc:
            i += frame_length
        else:
            sigaddlen = min(frame_shift_length, len(subsig))
            retsig[new_siglen:new_siglen + sigaddlen] = subsig[:sigaddlen]
            new_siglen += sigaddlen
            i += frame_shift_length
    retsig = retsig[:new_siglen]
    if is_unsigned:
        retsig = retsig + typeinfo.max / 2
    return retsig.astype(orig_dtype), fs
