from classifaudio.models.gmix import GMix, GMixInference
import glob


class Speaker(GMix):
    def __init__(self, name, components=32, max_iter=200, n_init=2):
        super().__init__(name, components, max_iter, n_init)

    @property
    def name(self):
        return self.label


class SpeakersModel(GMixInference):
    def __init__(self, speakers=None):
        speakers = speakers or []
        super().__init__(speakers)

    def add_speaker(self, speaker, replace=False):
        assert isinstance(speaker, Speaker)
        if speaker.name in self.registered_speakers:
            if not replace:
                raise ValueError("speaker exists")
            self.gmixs[self.registered_speakers.index(speaker.name)] = speaker
        else:
            self.gmixs.append(speaker)

    def load_speakers_from_folder(self, directory):
        if directory[-1] != "/":
            directory += "/"
        model_files = glob.glob(directory + "*.gmix")
        self.gmixs = [Speaker.load(a) for a in model_files]

    def save_speakers_to_folder(self, directory):
        for s in self.speakers:
            s.save(directory)

    @property
    def registered_speakers(self):
        return [s.name for s in self.speakers]

    @property
    def speakers(self):
        return self.gmixs

    def verify_speaker(self, voice_path, claimed_speaker, threshold=0.5):
        try:
            return self.verify(voice_path, claimed_speaker, threshold)
        except Exception:
            raise SpeakerNotFoundException()

    def predict_speaker(self, voice_path):
        return self.predict(voice_path)


class SpeakerNotFoundException(Exception):
    pass

