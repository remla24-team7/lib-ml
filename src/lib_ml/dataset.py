import os
import joblib
from pathlib import Path

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def load_dataset_file(path):
    with open(path, "r", encoding="utf-8") as fp:
        rows = [line.strip().split("\t") for line in fp.readlines()]
    raw_y, raw_x = list(zip(*rows))
    return raw_x, raw_y


def preprocess_dataset(dataset_dir, outputs_dir, sequence_length):
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token="-n-")
    encoder = LabelEncoder()

    raw_x_train, raw_y_train = load_dataset_file(Path(dataset_dir) / "train.txt")
    raw_x_val, raw_y_val = load_dataset_file(Path(dataset_dir) / "val.txt")
    raw_x_test, raw_y_test = load_dataset_file(Path(dataset_dir) / "test.txt")

    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

    def preprocess(x):
        return pad_sequences(tokenizer.texts_to_sequences(x), maxlen=sequence_length)

    x_train = preprocess(raw_x_train)
    x_val = preprocess(raw_x_val)
    x_test = preprocess(raw_x_test)

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    os.makedirs(outputs_dir, exist_ok=True)

    joblib.dump(x_train, Path(outputs_dir) / "x_train.joblib")
    joblib.dump(x_val, Path(outputs_dir) / "x_val.joblib")
    joblib.dump(x_test, Path(outputs_dir) / "x_test.joblib")

    joblib.dump(y_train, Path(outputs_dir) / "y_train.joblib")
    joblib.dump(y_val, Path(outputs_dir) / "y_val.joblib")
    joblib.dump(y_test, Path(outputs_dir) / "y_test.joblib")

    joblib.dump(tokenizer, Path(outputs_dir) / "tokenizer.joblib")
    joblib.dump(encoder, Path(outputs_dir) / "encoder.joblib")
