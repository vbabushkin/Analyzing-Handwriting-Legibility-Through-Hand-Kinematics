import gc
import os
import sys
import pickle

import numpy as np
import shap
import tensorflow as tf

import config

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import path_variables
import utilities
from model import Attention

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# Constants
RANDOM_STATE = config.RANDOM_STATE
N_FOLDS = config.N_FOLDS
OPT_WIN = config.OPT_WIN
OPT_OVR = config.OPT_OVR
overlap = int(OPT_WIN * OPT_OVR)
n = config.N_SAMPLES  # samples from train and test to calculate Shapley values

# Allow GPU memory to grow dynamically
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def calc_shapley_values(expert):
    for ifold in range(N_FOLDS):
        gc.enable()
        modelName = "sh_model_fold_" + str(ifold) + "_exp_" + str(expert) + "_all.h5"
        model = tf.keras.models.load_model(path_variables.MODELS_PATH + modelName, custom_objects={'CustomLayer': Attention})
        print(model.summary())

        with open(path_variables.RESULTS_PATH + 'sh_total_res_train_test_fold_' + str(ifold) + '_exp_' + str(
                expert) + '_all.pickle',
                  'rb') as handle:
            (X_train, y_train, X_test, y_test) = pickle.load(handle)
        label_binarizer = LabelBinarizer()

        # Check the performance of the model
        yhat = model.predict(X_test)
        label_binarizer.fit(y_train)
        yhat1 = label_binarizer.inverse_transform(yhat)

        reportStr = classification_report(y_test, yhat1, output_dict=False)
        print(reportStr)

        cm = confusion_matrix(y_test, yhat1, labels=np.unique(yhat1))
        print(cm)

        ##########################################################################################################################
        #  combine across classes (automate)
        ##########################################################################################################################
        #  Applying SHAP to  training set will help to inspect the ML model, and better understand the model's decision-making
        #  process ("opening up the black box"). However, it is only by applying SHAP to  testing set that  will be able to
        #  figure out how the features impact the model's generalization performance, so it is recommended applying SHAP
        #  to the testing set.
        # we look for only the class that coincides with the true label.
        # select a set of background examples to take an expectation over
        randomIdxTrain = np.random.choice(X_train.shape[0], n, replace=False)
        randomIdxTest = np.random.choice(X_test.shape[0], n, replace=False)
        background = X_train[randomIdxTrain]
        testLabels = y_test[randomIdxTest]

        print(model.layers[-1].output.name)  # print out the layer's name
        # passing tensors directly
        explainer = shap.DeepExplainer(
            (model.layers[0].input, model.layers[-1].output), background  # last layer output of the model
        )
        shap_values = explainer.shap_values(X_test[randomIdxTest], check_additivity=False)

        with open(path_variables.RESULTS_PATH + 'all_shap_values_' + modelName[:-3] + '.pickle', 'wb') as handle:
            pickle.dump((explainer.expected_value, X_test[randomIdxTest], testLabels, shap_values), handle)

        gc.collect()
        # Reset TensorFlow default graph (for TF1 compatibility)
        if hasattr(tf.compat.v1, 'reset_default_graph'):
            tf.compat.v1.reset_default_graph()

        gc.disable()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calc_shap_values.py <expert>")
        sys.exit(1)

    expert_arg = int(sys.argv[1])
    if not utilities.all_results_shap_values_exist(expert_arg):
        if not utilities.all_results_model_eval_exist(expert_arg, "all"):
            print(
                f"The files necessary for calculating Shapley values for expert {expert_arg} are missing in RESULTS folder.Please run model_eval_cv {expert_arg}.")
        else:
            print(
                f"Calculating Shapley values for expert {expert_arg} on model with\noptimal window {config.OPT_WIN}\noptimal overlap {config.OPT_OVR}")
            calc_shapley_values(expert_arg)
            print(f"Calculating Shapley values for expert {expert_arg} is complete. Results are saved.")
    else:
        print(f"Shapley values for expert {expert_arg} are already calculated. Skipping search.")