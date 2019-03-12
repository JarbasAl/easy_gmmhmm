import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
from scipy import fft
from python_speech_features import mfcc
from python_speech_features import delta


class FeaturesExtractor(object):
    """
    Helper Class to extract features from audio files
    """

    @staticmethod
    def extract(audio_path):
        """
        Utility wrapper around python_speech_features.mfcc

          Args:
            audio_path (str): path to load wav file from

        Returns:
            mffc_features
        """
        rate, audio = read(audio_path)
        return mfcc(audio, rate)

    @staticmethod
    def extract_fft_features(audio_path):
        """
        Utility wrapper around scipy.fft

        Args:
            audio_path (str): path to load wav file from

        Returns:
            fft_features
        """
        sample_rate, song_array = read(audio_path)
        fft_features = abs(fft(song_array[:30000]))
        return fft_features

    @staticmethod
    def extract_mfcc_features(audio_path, winlen=0.05, winstep=0.01, numcep=5,
                              nfilt=30, nfft=512, appendEnergy=True,
                              scale=True, deltas=True):
        """
         Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.

        Args:
            audio_path (str) : path to wave file without silent moments.
            winlen (float) : The length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            winstep (float) : The step between successive windows in seconds. Default is  0.01s (10 milliseconds)
            numcep (int) : The number of cepstrum to return. Default 13.
            nfilt (int) : The number of filters in the filterbank. Default is 26.
            nfft (int) : The FFT size. Default is 512.
            appendEnergy (bool) : If true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
            scale (bool) : If true, Center to the mean and component wise scale to unit variance.
            deltas (bool) : If true, combine with MFCC deltas and the MFCC double deltas.

        Returns:
            (array) : Extracted features matrix.

        """
        rate, audio = read(audio_path)
        return FeaturesExtractor. \
            extract_mfcc_features_from_data(audio, rate, winlen, winstep,
                                            numcep, nfilt, nfft,
                                            appendEnergy, scale, deltas)

    @staticmethod
    def extract_mfcc_features_from_data(audio, rate, winlen=0.05,
                                        winstep=0.01, numcep=5, nfilt=30,
                                        nfft=512, appendEnergy=True,
                                        scale=True, deltas=True):
        """
         Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.

        Args:
            audio (bytes) : wav data
            rate (int) : sample rate
            winlen (float) : The length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            winstep (float) : The step between successive windows in seconds. Default is  0.01s (10 milliseconds)
            numcep (int) : The number of cepstrum to return. Default 13.
            nfilt (int) : The number of filters in the filterbank. Default is 26.
            nfft (int) : The FFT size. Default is 512.
            appendEnergy (bool) : If true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
            scale (bool) : If true, Center to the mean and component wise scale to unit variance.
            deltas (bool) : If true, combine with MFCC deltas and the MFCC double deltas.

        Returns:
            (array) : Extracted features matrix.

        """
        mfcc_feature = mfcc(
            # The audio signal from which to compute features.
            audio,
            # The samplerate of the signal we are working with.
            rate,
            winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt,
            nfft=nfft, appendEnergy=appendEnergy)

        if scale:
            mfcc_feature = preprocessing.scale(mfcc_feature)
        if deltas:
            deltas = delta(mfcc_feature, 2)
            double_deltas = delta(deltas, 2)
            combined = np.hstack((mfcc_feature, deltas, double_deltas))
            return combined
        return mfcc_feature

    def collect_features(self, files):
        """
        Collect audio features from various files (of the same label).
        Args:
            files (list) : List of file paths.
        Returns:
            (array) : Extracted features matrix.
        """
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%5s %10s" % ("PROCESSING ", file))
            # extract MFCC & delta MFCC features from audio
            vector = FeaturesExtractor.extract_mfcc_features(file)
            # stack the features
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
        return features
