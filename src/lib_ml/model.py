import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Model:
    def __init__(self, model_path, tokenizer_path, encoder_path, sequence_length):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.encoder = joblib.load(encoder_path)
        self.sequence_length = sequence_length

    def preprocess(self, texts):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.sequence_length)

    def predict(self, texts):
        preprocessed = self.preprocess(texts)
        predictions = self.model.predict(preprocessed)
        return self.encoder.inverse_transform(predictions.round().astype(int))
