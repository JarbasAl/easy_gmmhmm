from classifaudio.models.gmix import GMix, GMixInference


class Gender(GMix):
    def __init__(self, name, components=32, max_iter=200, n_init=2):
        super().__init__(name, components, max_iter, n_init)

    @property
    def name(self):
        return self.label


class GendersModel(GMixInference):
    def __init__(self, speakers):
        super().__init__(speakers)

    @property
    def registered_genders(self):
        return [s.name for s in self.genders]

    @property
    def genders(self):
        return self.gmixs

    def verify_gender(self, voice_path, claimed_gender, threshold=0.5):
        try:
            return self.verify(voice_path, claimed_gender, threshold)
        except Exception:
            raise GenderNotFoundException()

    def predict_gender(self, voice_path):
        return self.predict(voice_path).name


class GenderNotFoundException(Exception):
    pass


if __name__ == "__main__":
    from os.path import join, dirname
    train = False
    save = True

    #male_path = join(dirname(__file__), "models", "male.gmix")
    #female_path = join(dirname(__file__), "models", "female.gmix")
    model_path = join(dirname(__file__), "models", "genders_gmix.pkl")
    if train:
        male_samples = join(dirname(__file__), "pygender", "train_data", "youtube",
                            "male")
        female_samples = join(dirname(__file__), "pygender", "train_data",
                              "youtube", "female")

        male = Gender("male").train_from_directory(male_samples)
        female = Gender("female").train_from_directory(female_samples)
        model = GendersModel([male, female])
        if save:
            model.save(model_path)
            #male.save(male_path)
            #female.save(female_path)
    else:
        #male = Gender.load(male_path)
        #female = Gender.load(female_path)
        #model = GendersModel([male, female])
        model = GendersModel.load(model_path)
        male = model.genders[0]

    print(model.registered_genders)
    female_test = join(dirname(__file__), "pygender", "test_data",
                       "AudioSet", "female_clips", "0--nzZoE_Ho.wav")
    male_test = join(dirname(__file__), "pygender", "test_data",
                       "AudioSet", "male_clips", "0-aEZiPFkzE.wav")
    print(male.score(male_test), male.score(female_test))
    print(model.predict_gender(female_test))
    print(model.verify_gender(female_test, "female"))
