from classifaudio.models.logreg import LogRegInference


class GendersModel(LogRegInference):
    def __init__(self):
        super().__init__(name="genders", labels=["male", "female"])

    @property
    def registered_genders(self):
        return self.labels

    def verify_gender(self, voice_path, claimed_gender, threshold=0.5):
        try:
            return self.verify(voice_path, claimed_gender, threshold)
        except Exception:
            raise GenderNotFoundException()

    def predict_gender(self, voice_path):
        return self.predict(voice_path)


class GenderNotFoundException(Exception):
    pass


if __name__ == "__main__":
    from os.path import join, dirname
    train = False
    save = True
    model_path = join(dirname(__file__), "models", "genders_logreg.logreg")
    if train:
        samples = join(dirname(__file__), "pygender", "train_data", "youtube")


        model = GendersModel().train_from_directory(samples)
        if save:
            model.save(model_path)
    else:
        model = GendersModel.load(model_path)

    print(model.registered_genders)
    female_test = join(dirname(__file__), "pygender", "test_data",
                       "AudioSet", "female_clips", "0--nzZoE_Ho.wav")
    male_test = join(dirname(__file__), "pygender", "test_data",
                       "AudioSet", "male_clips", "0-aEZiPFkzE.wav")
    print(model.predict_gender(male_test))
    print(model.verify_gender(female_test, "female"))
