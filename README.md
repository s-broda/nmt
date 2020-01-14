# Round Trip Loss for Machine Translation
Code repository for our report submitted for Deep Learning class autumn semester 2019.   

Repository includes code from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb (copyright The TensorFlow Authors).

### Required packages
Please install the required packages by running:
```
pip install -r requirements.txt
```

### Training a model
To train a model named model_test with default parameters run (with ```path2wd``` as the path to your working directory):
```
python train_transformer.py --train_dir path2wd --experiment_name model_test
```

To train a model on only 80% of the training data run
```
python train_transformer.py --train_dir path2wd --TRAIN_ON 80 --experiment_name model_test_80perc
```

### Evaluating a model
To evaluate a model named model_test_80perc run
```
python evaluate_transformer.py --train_dir path2wd --experiment_name model_test_80perc
```

### Creating backtranslation w model
To backtranslate one can only use a model that was not trained on all of the training set (params TRAIN_ON < 100)
To create backtranslation with a previously trained model_test_80perc run:
```
python backtrans_w_transformer.py --train_dir path2wd --experiment_name model_test_80perc
```
To train another model with the additional backtranslated training data run:
```
python train_transformer.py --train_dir path2wd 10 --include_backtrans_of_model model_test --experiment_name model_test_wBacktrans
```
