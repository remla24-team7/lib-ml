# lib-ml

[model.py](src/lib_ml/model.py) provides a `Model` class that depends on a `Tokenizer`, a Keras `Model`, and a `LabelEncoder` to make (text label) predictions.

[dataset.py](src/lib_ml/dataset.py) provides a `preprocess_dataset(dataset_dir, dest_dir)` method that writes artifacts to `dest_dir`.
