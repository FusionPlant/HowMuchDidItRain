0. Basic

0.1 console pipeline in windows
Use 
    python -u [PYTHON_FILENAME]
can get realtime pipeline output in window console.

1. Preprocess

1.0 Use preprocess.normal.py first, dask version is very slow in my env.

Command example
    python -u preprocess.normal.py

Input and output file path is hardcoded in file.

1.1 Format of preprocessing result

ID, Distance, Number of Observation
8 * Ref
Expected(for training data)

2. Prediction

Command example
    python -u predict.py
