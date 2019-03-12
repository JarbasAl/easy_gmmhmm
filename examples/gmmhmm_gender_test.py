from classifaudio.classifiers.gmmhmm import GMMHMMTester
from os.path import join, dirname


model_name = "gender_gmmhmm"
model_path = join(dirname(__file__), "models", model_name + ".pkl")
data_path = join(dirname(__file__), "pygender", "test_data", "AudioSet")


tester = GMMHMMTester(model_path)

# test single file
wav_file = join(data_path, "male_clips", "0-Rf6bTD5fs.wav")
predicted, probs = tester.predict_label(wav_file)
print("PREDICTED: %s" % predicted[0])
print("scores: {}".format(probs))

# test whole folder
from os import listdir

print("testing male clips")
wrongs = 0
for wav_file in listdir(join(data_path, "male_clips")):
    wav_file = join(data_path, "male_clips", wav_file)
    predicted, probs = tester.predict_label(wav_file)
    if predicted[0] != "male":
        wrongs += 1
print("wrong male classifications:", wrongs)

print("testing female clips")
wrongs = 0
for wav_file in listdir(join(data_path, "female_clips")):
    wav_file = join(data_path, "female_clips", wav_file)
    predicted, probs = tester.predict_label(wav_file)
    if predicted[0] != "male":
        wrongs += 1
print("wrong female classifications:", wrongs)