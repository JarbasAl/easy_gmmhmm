# easy GMM-HMM
[![Donate with Bitcoin](https://en.cryptobadges.io/badge/micro/1QJNhKM8tVv62XSUrST2vnaMXh5ADSyYP8)](https://en.cryptobadges.io/donate/1QJNhKM8tVv62XSUrST2vnaMXh5ADSyYP8)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/jarbasai)
<span class="badge-patreon"><a href="https://www.patreon.com/jarbasAI" title="Donate to this project using Patreon"><img src="https://img.shields.io/badge/patreon-donate-yellow.svg" alt="Patreon donate button" /></a></span>
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/JarbasAl)

GMM-HMM (Hidden markov model with Gaussian mixture emissions) for sound recognition and other uses

[read the docs](https://JarbasAl.github.io/easy_gmmhmm/)


# install

    pip install numpy
    pip install scipy
    pip install python_speech_features
    pip install hmmlearn
    pip install git+https://github.com/JarbasAl/easy_gmmhmm
    
# train

takes in the directory containing training data as raw wavfiles within folders named according to label and extracts MFCC feature vectors from them,
accepts a configuration for each in terms of number of states for HMM and number of mixtures in the Gaussian Model and then trains a set of GMMHMMs,
one for each label.
    
Params:
-data_path: Path to the training wav files. Each folder in this path is a label and must NOT be empty.
-model_path: Path to store the generated pickle files in.
    
    from easy_gmmhmm import train
    
    model_path = "models"
    data_path = "data"
    train(data_path, model_path)

# test

    from easy_gmmhmm import predict_label
    import time

    start = time.time()
    wav_file = "test.wav"
    model_path = "models"
    predicted, probs = predict_label(wav_file, model_path)
    print("PREDICTED: %s" % predicted[0])
    print("scores: {}".format(probs))
    print("elapsed", time.time() - start)
