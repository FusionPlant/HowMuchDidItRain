Basic
=====

* console pipeline in windows
Use 
```
    python -u [PYTHON_FILENAME]
```
can get realtime pipeline output in window console.

Preprocess
=====

* Use preprocess.normal.py first, dask version is very slow in my env.

Command example
```
    python -u preprocess.normal.py
```

Input and output file path is hardcoded in file.

* Format of preprocessing result

ID, Distance, Number of Observation
8 * Ref
12 features(average of observation set)
Expected(for training data)

Prediction
=====

Command example
```
    python -u predict.py
```
