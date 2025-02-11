import pickle
import sys
import warnings
from sys import platform

import shap
import tensorflow as tf
from scipy import stats
from statannot import add_stat_annotation

import config

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)
import matplotlib.pyplot as plt
import sklearn.preprocessing
import os
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import path_variables
# Suppress warnings
warnings.filterwarnings("ignore")
import seaborn as sns
# Matplotlib settings
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# load the dataset
def load_data():
    """
    Load kinematics data and labels.
    """
    labels = pd.read_csv(path_variables.MAIN_DATA_FOLDER + "avg_legibility_score.csv")
    labels = labels[["subj", "rep"]]
    labels["sub_id"] = labels["subj"]

    with open(path_variables.MAIN_DATA_FOLDER + 'total_dataset_raw.pickle', 'rb') as handle:
        X, Y, subjInfo = pickle.load(handle)

    return X, Y, subjInfo


# a function to check for the required files in path_variables.RESULTS before running param_search_overlap.param_search_overlap().
def all_results_overlap_search_exist(expert):
    """Check if results exist for a given expert."""
    required_files = [f"param_search_ovr_expert_{expert}_run_{run}.csv" for run in range(5)]
    return all(os.path.isfile(os.path.join(path_variables.RESULTS_PATH, file)) for file in required_files)

# a function to check for the required files in path_variables.RESULTS before running param_search_win.param_search_win().
def all_results_win_search_exist(expert):
    """Check if results exist for a given expert."""
    required_files = [f"param_search_win_expert_{expert}_run_{run}.csv" for run in range(5)]
    return all(os.path.isfile(os.path.join(path_variables.RESULTS_PATH, file)) for file in required_files)

# a function to check for the required files in path_variables.RESULTS before running model_eval_cv.
def  all_results_model_eval_exist(expert, mode):
    """Check if results exist for a given expert."""
    model_files = [f"sh_model_fold_{ifold}_exp_{expert}_{mode}.h5" for ifold in range(5)]
    pickle_files = [f"sh_total_res_train_test_fold_{ifold}_exp_{expert}_{mode}.pickle" for ifold in range(5)]
    required_files = [os.path.join(path_variables.MODELS_PATH, file) for file in model_files]
    required_files.extend(os.path.join(path_variables.RESULTS_PATH, file) for file in pickle_files)
    required_files.append(os.path.join(path_variables.RESULTS_PATH,f"sh_total_res_expert_{expert}_{mode}.csv"))
    return all(os.path.isfile(file) for file in required_files)

# a function to check for the required files in path_variables.RESULTS before running calc_shapley_values.
def  all_results_shap_values_exist(expert):
    """Check if results exist for a given expert."""
    required_files = [f"all_shap_values_sh_model_fold_{ifold}_exp_{expert}_all.pickle" for ifold in range(5)]
    return all(os.path.isfile(os.path.join(path_variables.RESULTS_PATH, file)) for file in required_files)

def plot_shap_values(mean_shap_valuesq, data_for_prediction,modelName, ifold, n, custom_palette, class_type = "all"):
    fig = plt.figure()
    shap.summary_plot(mean_shap_valuesq.T, data_for_prediction, plot_type='bar', plot_size=(8.5, 6), max_display=12,
                      color=custom_palette[3],show=False)  # for over channels
    fig = plt.gcf()  # Get current figure
    fig.set_size_inches(15, 8)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=18)
    plt.tight_layout()
    plt.savefig(
        path_variables.FIGURES_PATH + 'shap_summary_plot_expert_' + modelName[:-3] + "_fold_" + str(ifold) + '_n_' + str(n) + "_" + class_type+'.pdf')


# outputs the data on GPU usage
def print_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                details = tf.config.experimental.get_memory_info(gpu.name)
                print(f"GPU Memory Allocated: {details['current']} / {details['peak']}")
        except Exception as e:
            print(f"Error getting GPU memory info: {e}", file=sys.stderr)

# reads and preprocesses the raw kinemtaics data
# datapath -- path to the raw kinematics data
# labelDf -- dataframe where all the labels are stored
# applyFilter -- boolean value if we want to clean the data with median filter
# MED_FILTER_WIN -- the length of median filter
def load_multilabel_kinematics_data(datapath, labelDf, applyFilter, MED_FILTER_WIN = 20):
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []
    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*.csv")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)

    X = []
    Y = []
    subjInfo = []

    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[2])
            repNum = int(currentFile.split('/')[-1].split('_')[3][-1])
        df = pd.read_csv(currentFile)
        colsToRemove = ['handId', 'sec', 'min', 'hour', 'lifetimeOfThisHandObject', 'confidence']
        df = df.drop(colsToRemove, axis=1)
        featuresNames = df.columns.to_numpy()
        tmpX = df[featuresNames].to_numpy().astype(np.float32)
        tmpX = tmpX[2:-2, :]

        tmpDf = labelDf.set_index(['subj', 'rep'])
        label1 = tmpDf.at[(subjNum, repNum), "legibility_score_1"].astype(int)
        label2 = tmpDf.at[(subjNum, repNum), "legibility_score_2"].astype(int)
        label3 = tmpDf.at[(subjNum, repNum), "legibility_score_3"].astype(int)
        Y.append([label1,label2,label3])

        w = np.zeros(shape=tmpX.shape)
        # apply median filter
        if applyFilter:
            for j in range(tmpX.shape[1]):
                w[:, j] = median_filter(tmpX[:, j], MED_FILTER_WIN)
            X.append(w)
        else:
            X.append(tmpX)
        subjInfo.append([subjNum, repNum, label1, label2, label3])
    return X, Y, subjInfo


# for preprocessing the raw kinematics data
def preprocess():
    if(os.path.isfile(path_variables.MAIN_DATA_FOLDER + 'total_dataset_raw.pickle')):
        print("The data are already preprocessed, file total_dataset_raw.pickle is in "+path_variables.MAIN_DATA_FOLDER+ " folder." )
    else:
        print("Loading and preprocessing the raw data...")
        labels = pd.read_csv(path_variables.MAIN_DATA_FOLDER + "avg_legibility_score.csv")
        labels["sub_id"] = labels["subj"]
        # load main data
        # VERY IMPORTANT
        # do not filter
        ########################################################################################################################
        # get train and test sets split into windows
        ########################################################################################################################
        (X, Y, subjInfo) = load_multilabel_kinematics_data(path_variables.KINEMATICS_RAW_DATA, labels, False)
        subjInfo = np.array(subjInfo)
        Y = np.array(Y)

        with open(path_variables.MAIN_DATA_FOLDER + 'total_dataset_raw.pickle', 'wb') as handle:
            pickle.dump((X, Y, subjInfo), handle, protocol=4)

# for splitting into sliding windows of length L with overlap ov
# a - array,
# L -length of the window
# ov - overlap
def get_strides(a, L, ov):
    if a.shape[0]<L:
        out = np.pad(a, ((0, L-a.shape[0]),(0,0)), 'constant', constant_values=0)
        out = np.expand_dims(out, 0)
    else:
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        tmpA = np.zeros((L, a.shape[1]))
        tmpA[:L - (i + 2 * L - ov - a.shape[0]), :] = a[i + L - ov - 1:a.shape[0] - 1, :]
        out.append(tmpA)
    return np.array(out)

# split into train test for each fold
# currentTrainSubjInfo contains train indices for train set of current fold
# currentTestSubjInfo contains test indices for test set of current fold
# subjInfo contains main dataset indices before splitting
# X and Y are main dataset X and lables Y before splitting
def fold_train_test_windows(X,Y,currentTrainSubjInfo,currentTestSubjInfo, subjInfo, winSize, overlap, scale = True):
    # create training set
    for i in range(len(currentTrainSubjInfo)):
        currentTrainSubj = currentTrainSubjInfo[i, 0]
        currentTrainPar = currentTrainSubjInfo[i, 1]
        currentTrainIdx = np.where((subjInfo[:, 0] == currentTrainSubj) & (subjInfo[:, 1] == currentTrainPar))[0]
        if i == 0:
            foldTrainIdx = currentTrainIdx
        else:
            foldTrainIdx = np.hstack((foldTrainIdx, currentTrainIdx))

    pre_y_train = Y[foldTrainIdx]
    pre_X_train = [X[idx] for idx in foldTrainIdx]

    # create testing set
    for i in range(len(currentTestSubjInfo)):
        currentTestSubj = currentTestSubjInfo[i, 0]
        currentTestPar = currentTestSubjInfo[i, 1]
        currentTestIdx = np.where((subjInfo[:, 0] == currentTestSubj) & (subjInfo[:, 1] == currentTestPar))[
            0]
        if i == 0:
            foldTestIdx = currentTestIdx  # these are already testing indices that include lines
        else:
            foldTestIdx = np.hstack((foldTestIdx, currentTestIdx))

    pre_y_test = Y[foldTestIdx]
    pre_X_test = [X[idx] for idx in foldTestIdx]

    # now for each line split the data with the sliding window
    trainSet_X = []
    trainSet_Y = []
    testSet_X = []
    testSet_Y = []

    for i in range(len(pre_X_train)):
        tmpX = pre_X_train[i]
        # sliding window
        a = get_strides(tmpX, winSize, overlap)
        trainSet_X.append(a)
        trainSet_Y.append(np.repeat(pre_y_train[i], a.shape[0]))
    X_train = np.concatenate(trainSet_X, axis = 0)
    y_train = np.concatenate(trainSet_Y, axis = 0)
    print("Finished stacking X_train")

    for i in range(len(pre_X_test)):
        tmpX = pre_X_test[i]
        # sliding window
        a = get_strides(tmpX, winSize, overlap)
        testSet_X.append(a)
        testSet_Y.append(np.repeat(pre_y_test[i], a.shape[0]))
    X_test = np.concatenate(testSet_X, axis = 0)
    y_test = np.concatenate(testSet_Y, axis = 0)
    print("Finished stacking X_test")

    if scale:
        # scale
        s = sklearn.preprocessing.StandardScaler()
        for i in range(X_train.shape[0]):
            X_train[i] = s.fit_transform(X_train[i])

        for i in range(X_test.shape[0]):
            X_test[i] = s.fit_transform(X_test[i])
    return(X_train,y_train,X_test, y_test)


def plot_avg_cm(cmPerFold,path, filename,classNames=None ):
    avgCm = np.mean(np.array(cmPerFold), axis=0)
    if classNames is None:
        classNames = [0,1,2]
    df_cm = pd.DataFrame(avgCm, index=classNames, columns=classNames)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in avgCm.flatten()]
    div = np.divide(avgCm.T, np.sum(avgCm, 1)).T.flatten()
    res = np.argwhere(np.isnan(div)).flatten()
    div[res] = 0
    group_percentages = ["{0:.2%}".format(value) for value in div]  # cm.flatten()/np.sum(cm)]
    labels = ["{}\n{}".format(v1, v2) for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(len(classNames), len(classNames))

    plt.figure(figsize=(15, 8))
    sns.set(font_scale=1.4)  # for label size
    np.min(avgCm)
    div = (np.divide(avgCm.T, np.sum(avgCm, 1)).T) * 100
    res = np.argwhere(np.isnan(div))
    for i, j in res:
        div[i, j] = 0
    ax = sns.heatmap(div,
                     cbar_kws={'ticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}, vmin=0, vmax=100.0,
                     annot=labels, annot_kws={"size": 22}, fmt='', cmap="Blues")  # font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(axis='x', which='major', pad=-3)
    ax.figure.axes[-1].tick_params(labelsize=26)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
    ax.set_yticklabels(classNames, rotation=0, fontsize="26", va="center")
    ax.set_xticklabels(classNames, rotation=0, fontsize="26", ha="center")
    ax.set_ylabel("True Label", fontsize="40")
    ax.set_xlabel("Predicted Label", fontsize="40")
    plt.tight_layout()
    plt.savefig(path + filename + '.pdf')


# plot statistical results (boxplots, low and high legibility)
def plot_stat_analysis_results(df_feature, feature_name, label,low_legibility_data, high_legibility_data, isNormal, isEqualStd):
    sns.set(
        style="whitegrid",
        font_scale=1.5,
        # style="ticks",  # The 'ticks' style
        rc={'axes.edgecolor': '.15', 'ytick.color': '.15', 'ytick.left': True, "figure.figsize": (3.5, 5.66),
            # width = 6, height = 4
            })  # Axes colour
    # plot
    fig = plt.figure()
    yl = feature_name
    xl = "legibility"
    order = ["low", "high"]
    ax = sns.boxplot(data=df_feature, x=xl, y=yl, order=order,
                     width=0.5,  # The width of the boxes
                     color="skyblue",  # Box colour
                     linewidth=2,  # Thickness of the box lines
                     showfliers=False,
                     showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": ".15"})

    ax.set_xlabel("", fontsize=2)
    # Set the x axis label and font size
    ax.set_ylabel(label, fontsize=24)
    ax.grid(False)
    plt.gca().set_xticklabels(order, size=24)
    plt.ylim([-0.01, 1.01])
    if isNormal and isEqualStd:
        t, p = stats.ttest_ind(low_legibility_data[:, 0], high_legibility_data[:, 0])
        print("%s t-test:      n1=%g, M1 = %g   n2=%g, M2 = %g      t = %g  p = %g" % (
            feature_name, len(low_legibility_data[:, 0]), np.median(high_legibility_data[:, 0]), len(high_legibility_data[:, 0]), np.median(low_legibility_data[:, 0]), t, p))
        if p <= 1.00e-02:
            pval = 0.01
            pval_str = "p < 0.01"
        else:
            pval = p
            pval_str = "p = " + str(round(p, 3))
        a = add_stat_annotation(ax, data=df_feature, x=xl, y=yl, order=order, perform_stat_test=False, pvalues=[pval],
                                box_pairs=[("high", "low")], pvalue_format_string="{.4}", text_annot_custom=[pval_str],
                                test=None, text_format="simple", loc='outside', verbose=2)
        sns.despine(offset=5, trim=True, bottom=True)
        plt.tight_layout()
        plt.savefig(path_variables.FIGURES_PATH + 'win_time_avg_' + feature_name + '_all_t_test.pdf')
    else:
        # Use scipy.stats.ttest_ind.
        t, p = stats.mannwhitneyu(low_legibility_data[:, 0], high_legibility_data[:, 0])
        print("%s MW_U:      n1=%g, M1 = %g   n2=%g, M2 = %g      t = %g  p = %g" % (
            feature_name, len(low_legibility_data[:, 0]), np.median(low_legibility_data[:, 0]), len(high_legibility_data[:, 0]), np.median(high_legibility_data[:, 0]), t, p))
        if p <= 1.00e-02:
            pval = 0.01
            pval_str = "p < 0.01"
        else:
            pval = p
            pval_str = "p = " + str(round(p, 3))
        a = add_stat_annotation(ax, data=df_feature, x=xl, y=yl, order=order, perform_stat_test=False, pvalues=[0.0001],
                                box_pairs=[("high", "low")], pvalue_format_string="{.4}", text_annot_custom=[pval_str],
                                test=None, text_format="simple", loc='outside', verbose=2)

        sns.despine(offset=5, trim=True, bottom=True)
        plt.tight_layout()
        plt.savefig(path_variables.FIGURES_PATH + 'win_time_avg_' + feature_name + '_all_mw_test.pdf')


def conduct_stat_tests(feature_name, label, val_low, val_high):
    # plot for pressure variability
    vals1 = np.squeeze(val_low)
    vals2 = np.squeeze(val_high)

    low_legibility_data = np.vstack((vals1, np.zeros(vals1.shape[0]))).T
    high_legibility_data = np.vstack((vals2, np.ones(vals2.shape[0]))).T

    # t-test assumptions:
    # Independence of observations
    # Normality
    # Equal variances

    # based on D’Agostino and Pearson’s test that combines skew and kurtosis to produce an omnibus test of normality.
    res1 = stats.mstats.normaltest(vals1)
    res2 = stats.mstats.normaltest(vals2)

    isNormal = True
    if res1.pvalue < 0.05:
        print("low legibility sample is not normal")
        isNormal = False
    if res2.pvalue < 0.05:
        print("high legibility sample is not normal")
        isNormal = False

    # check equal variances assumption
    test_statistic, p_value = stats.bartlett(vals1, vals2)
    isEqualStd = True
    if p_value < 0.05:
        print("the groups do not have equal variances")
        isEqualStd = False
    else:
        print("the groups variances are equal")

    feature_data = np.vstack((low_legibility_data, high_legibility_data))
    feature_data[:, 0] = (feature_data[:, 0] - min(feature_data[:, 0])) / (
                max(feature_data[:, 0]) - min(feature_data[:, 0]))
    df_feature = pd.DataFrame({feature_name: feature_data[:, 0], 'legibility': feature_data[:, 1]})
    df_feature["legibility"].iloc[np.where(df_feature["legibility"] == 0)] = "low"
    df_feature["legibility"].iloc[np.where(df_feature["legibility"] == 1)] = "high"

    plot_stat_analysis_results(df_feature, feature_name, label, low_legibility_data, high_legibility_data,
                                         isNormal, isEqualStd)

def plot_correlations(var_1, var_2, l_idx_subj, h_idx_subj, x_lab, y_lab, filename):
    all_vars = np.vstack((var_1, var_2))
    sorted_indices = np.argsort(all_vars[0])
    # Sort all rows based on the order of the first row
    all_vars_sorted = all_vars[:, sorted_indices]

    np.corrcoef((all_vars_sorted[0], all_vars_sorted[1]))[0, 1]
    l_idx_subj = np.array(l_idx_subj)
    h_idx_subj = np.array(h_idx_subj)
    other_idx = np.setdiff1d(np.arange(1, config.NUM_SUBJECTS+1), np.concatenate([l_idx_subj, h_idx_subj]))

    # for low/high legibility
    fig = plt.figure(figsize=(8,5))
    plt.scatter(all_vars_sorted[0, other_idx - 1], all_vars_sorted[1, other_idx - 1], marker='o', facecolors='none',
                edgecolor='k', label="other", alpha=0.5)
    plt.scatter(all_vars_sorted[0, l_idx_subj - 1], all_vars_sorted[1, l_idx_subj - 1], c='b', marker='o',
                label="low")
    plt.scatter(all_vars_sorted[0, h_idx_subj - 1], all_vars_sorted[1, h_idx_subj - 1], c='r', marker='o',
                label="high")
    correlation_coefficient = np.corrcoef((all_vars_sorted[0], all_vars_sorted[1]))[0, 1]
    plt.text(0.5, 0.9, f'correlation coefficient: {correlation_coefficient:.2f}',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14, color='k')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend(title="legibility", fontsize=12)
    plt.tight_layout()
    plt.savefig(path_variables.FIGURES_PATH+filename)