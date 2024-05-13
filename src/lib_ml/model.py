import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 200

class Model:
    def __init__(self, model_path, tokenizer_path, encoder_path):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.encoder = joblib.load(encoder_path)

    def preprocess(self, texts):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=SEQUENCE_LENGTH)

    def predict(self, texts):
        preprocessed = self.preprocess(texts)
        predictions = self.model.predict(preprocessed)
        # labels = self.encoder.inverse_transform(predictions)
        return None
