from easy_gmmhmm.classifiers.gmmhmm import GMMHMMTrainer
from os.path import join, dirname

# where to save the model
model_path = join(dirname(__file__), "models")
# path with folders, where folder name is a label and inside are .wav files
data_path = join(dirname(__file__), "pygender", "train_data", "youtube")
# name of this model, saved in model_path as .pkl
model_name = "gender_gmmhmm"

# config optional, if not provided is asked interactively
config = {"male": {"n_components": 10, "n_mix": 10},
          "female": {"n_components": 10, "n_mix": 10}}


trainer = GMMHMMTrainer(data_path, model_name, model_path, config)
trainer.train()
