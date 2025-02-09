import os
import pickle
import sys

import numpy as np
import shap
import tensorflow as tf

import config

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)

import path_variables
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utilities

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# Constants
RANDOM_STATE = config.RANDOM_STATE
N_FOLDS = config.N_FOLDS
OPT_WIN = config.OPT_WIN
OPT_OVR = config.OPT_OVR
overlap = int(OPT_WIN * OPT_OVR)
n = config.N_SAMPLES  # samples from train and test to calculate Shapley values
custom_palette = sns.color_palette("Blues", 5)

# Allow GPU memory to grow dynamically
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def plot_shapley_values(expert):
    mean_shap_valuesq_arr = []
    for ifold in range(N_FOLDS):
        print("Running fold " + str(ifold))

        modelName = "sh_model_fold_" + str(ifold) + "_exp_" + str(expert) + "_all.h5"

        with open(path_variables.RESULTS_PATH + 'all_shap_values_' + modelName[:-3] + '_all.pickle', 'rb') as handle:
            (explainer, X_test, testLabels, shap_values) = pickle.load(handle)

        data_for_prediction = pd.DataFrame(X_test[0], columns=config.features_117)

        shap_valuesq = np.array(shap_values)
        print(shap_valuesq.shape)
        # before
        shap_valuesq.shape  # (num_samples, num_timepoints, num_features, num_classes)
        shap_valuesq = np.transpose(shap_valuesq, (3, 0, 1, 2))

        shap_valuesq.shape  # (num_classes, num_samples, num_timepoints, num_features)

        # we select only those shapeley values that correspond to the true labels
        labelsClass = testLabels.astype(int)
        idxClass = labelsClass - 1  # check

        # here we store shapley values only for classes that correspond to true labels.
        uniqueClasses = np.unique(idxClass)
        shapValuesByClassLabel = []

        for k in range(n):
            shapValuesByClassLabel.append(shap_valuesq[np.where(uniqueClasses == idxClass[k])[0][0], k, :, :])

        shapValuesByClassLabel = np.array(shapValuesByClassLabel)
        # averaged shapley values across all 100 samples
        mean_shap_valuesq = np.swapaxes(np.mean(np.abs(shapValuesByClassLabel), axis=0), 0, 1)
        mean_shap_valuesq_arr.append(mean_shap_valuesq)
        utilities.plot_shap_values(mean_shap_valuesq, data_for_prediction, modelName, ifold, n, custom_palette)

    mean_shap_valuesq_arr = np.array(mean_shap_valuesq_arr)

    # normalize
    mean_shap_valuesq_arr_norm = []

    for i in range(5):
        mean_shap_valuesq_arr_norm.append((mean_shap_valuesq_arr[i, :, :] - np.min(mean_shap_valuesq_arr[i, :, :])) / (
                    np.max(mean_shap_valuesq_arr[i, :, :]) - np.min(mean_shap_valuesq_arr[i, :, :])))

    mean_shap_valuesq_arr_norm = np.array(mean_shap_valuesq_arr_norm)

    total_mean_shap_valuesq = np.mean(mean_shap_valuesq_arr_norm, axis=0)

    print(total_mean_shap_valuesq.shape)

    fig = plt.figure()
    # bar plot graph
    shap.summary_plot(total_mean_shap_valuesq.T, data_for_prediction, plot_type='bar', max_display=10,
                      color=custom_palette[3],show=False)
    # Capture the SHAP-generated figure
    fig = plt.gcf()  # Get current figure
    fig.set_size_inches(15, 8)  # Resize if needed
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel("normalized mean(|SHAP value|) (average impact on model output magnitude)", fontsize=18)
    plt.tight_layout()
    plt.savefig(
        path_variables.FIGURES_PATH + "shap_summary_plot_expert_n_" + str(n) + "_avg_across_folds_all_norm_n_" + str(n) + ".pdf")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_shapley_values.py <expert>")
        sys.exit(1)

    expert_arg = int(sys.argv[1])
    if not utilities.all_results_shap_values_exist(expert_arg):
        print(
            f"The files necessary for plotting Shapley values for expert {expert_arg} are missing in RESULTS folder.Please run calc_shapley_values.py {expert_arg}.")
    else:
        print(
            f"Calculating Shapley values for expert {expert_arg} on model with\noptimal window {config.OPT_WIN}\noptimal overlap {config.OPT_OVR}")
        plot_shapley_values(expert_arg)
        print(f"Calculating Shapley values for expert {expert_arg} is complete. Results are saved.")



