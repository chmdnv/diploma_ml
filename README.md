# Credit Risk Management Model
## Learning Project for Skillbox ML-junior
The project solved the problem of predicting risk of client's default, based on credit history data

Used data from https://www.kaggle.com/datasets/mikhailkostin/dlfintechbki<br>
<br>
3 prediction models were tested: LightGBM, HistGradientBoosting and a neural network implemented on Pytorch<br>
<br>
Final accuracy of the best model on the test sample:
0.7625 (roc-auc)

## Project content
**/data** -  project data (download from source)
* test_predict.csv - predicted values for test_data

**/jupyter** - jupyter-notebook files:
* collect - data loading and aggregation
* modeling - model selection (LightGBM, HistGradient)
* modeling_mlp - Pytorch neural network
* pipeline - creating a pipelines with the final model
* pipeline_test - test predict on the pipeline
* pipe_utils.py - utilities and classes for work of the pipelines
* features.pkl - list of final features of the model

**/model** - dumped model files
<br><br>
## Launch
Final pipeline **model/pipe_11.pkl**<br>
Import all the functions from jupyter/pipe_utils.py<br>
To deserialize the *pipe_11* model use dill (see example in *pipeline_test.ipynb*)
