import gc
import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import sys
import config
import path_variables
import utilities
from model import create_model

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
from tensorflow.python.framework import ops

# Constants
RANDOM_STATE = config.RANDOM_STATE
N_FOLDS = config.N_FOLDS
KERNEL = config.KERNEL
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR
DROPOUT = config.DROPOUT
LOSS_FN = config.LOSS_FN
NUM_EXPERTS = config.NUM_EXPERTS
NUM_SUBJECTS = config.NUM_SUBJECTS
OPT_WIN = config.OPT_WIN
OPT_OVR = config.OPT_OVR

# Allow GPU memory to grow dynamically
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model_eval_cv(expert, mode):
    X, Y, subjInfo = utilities.load_data()
    print(f"Evaluating model with expert {expert} labels for {mode}.")
    exp_subj_info = np.zeros((subjInfo.shape[0], 3))
    for k in range(1, NUM_SUBJECTS + 1):
        exp_subj_info[np.where(subjInfo[:, 0] == k)[0], 0:2] = subjInfo[np.where(subjInfo[:, 0] == k)[0], 0:2]
        exp_subj_info[np.where(subjInfo[:, 0] == k)[0], 2] = np.mean(
            subjInfo[np.where(subjInfo[:, 0] == k)[0], int(expert + 1)])

    Y = np.round(exp_subj_info[:, 2]).astype(int)
    exp_subj_info[:, 2] = np.round(exp_subj_info[:, 2]).astype(int)
    subjInfo = exp_subj_info
    num_classes = np.unique(Y).shape[0]
    #################################################################################################################
    #
    # split into test -train
    #
    #################################################################################################################
    # here we want to prepare fold so that testing and training set for each fold contain lines
    # of different paragraph (even for the same subject)
    # and later these folds will be modified to adapt to the lines for each subject's paragraph
    # i.e. we want the lines from same paragraph remain in either testing or in traing sets,
    # but not one line from the same paragraph is in training and another line is in testing set
    X_subj_par = subjInfo[:, :2]
    Y_subj_par = subjInfo[:, 2]  # stratify by labels

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    report_filename = path_variables.RESULTS_PATH + "sh_total_res_expert_" + str(expert) + "_" + str(mode)+ ".csv"
    utilities.print_gpu_usage()
    overlap = int(OPT_WIN * OPT_OVR)
    #################################################################################################################
    #
    # create a dataframe for storing the reports
    #
    #################################################################################################################
    if os.path.isfile(report_filename):
        reports_df = pd.read_csv(report_filename, index_col=False)
    else:
        reports_df = pd.DataFrame(
            columns=['WINDOW', 'OVERLAP', 'EPOCHS', 'LEARNING_RATE', 'BATCH_SIZE', 'DROPOUT', 'ACC',
                     'PRECISION',
                     'RECALL', 'F1', 'FOLD', 'AVG_ACC', 'AVG_PREC', 'AVG_RECALL', 'AVG_F1', 'TIME'])

    # Create cross-validation object
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(enumerate(kf.split(X_subj_par, Y_subj_par)))
    cmPerFold = []
    # Run the cross-validation
    reports_per_fold = []
    for ifold in range(N_FOLDS):
        gc.enable()
        start_time = time.time()
        print(f"Cross-validating model for fold {ifold} for expert {expert}")
        foldTrainIdx = folds[ifold][1][0]
        foldTestIdx = folds[ifold][1][1]
        currentTrainSubjInfo = subjInfo[foldTrainIdx, :]
        currentTestSubjInfo = subjInfo[foldTestIdx, :]

        ########################################################################################################################
        # get train and test sets split into windows
        ########################################################################################################################
        (X_train, y_train, X_test, y_test) = utilities.fold_train_test_windows(X, Y, currentTrainSubjInfo,
                                                                               currentTestSubjInfo,
                                                                               subjInfo,
                                                                               OPT_WIN, overlap, scale=True)
        if mode == "stylus":
            X_train = X_train[:, :, :7]
            X_test = X_test[:, :, :7]
        elif mode == "hand":
            X_train = X_train[:, :, 7:]
            X_test = X_test[:, :, 7:]

        n_features = X_train.shape[-1]
        y_train_orig = y_train
        y_test_orig = y_test
        print(
            f"Fold {ifold}: X_train size={X_train.shape}, y_train size={len(y_train_orig)}, X_test size={X_test.shape}, y_test size={len(y_test_orig)}")

        with open(path_variables.RESULTS_PATH + 'sh_total_res_train_test_fold_' + str(ifold) + '_exp_' + str(
                expert) + "_" + str(mode) + '.pickle', 'wb') as handle:
            pickle.dump((X_train, y_train_orig, X_test, y_test_orig), handle)
        print("Files are saved")

        a = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

        X1_train, y1_train = ros.fit_resample(a, y_train)
        X1_train = X1_train.reshape(X1_train.shape[0], OPT_WIN, n_features)

        # Create an instance of One-hot-encoder
        label_binarizer = LabelBinarizer()
        y_train = label_binarizer.fit_transform(y1_train)
        y_test = label_binarizer.fit_transform(y_test)

        model, sess = create_model(X1_train.shape[1:], KERNEL, DROPOUT, LR, LOSS_FN, num_classes)
        model.fit(X1_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

        modelName =  "sh_model_fold_" + str(ifold) + '_exp_' + str(expert) + "_" + str(mode)+".h5"
        tf.keras.models.save_model(model,path_variables.MODELS_PATH + modelName)

        y_pred = model.predict(X_test)

        sess.close()
        ops.reset_default_graph()
        tf.keras.backend.clear_session()
        del X_train, X_test, X1_train, model, a
        gc.collect()

        y_pred = label_binarizer.inverse_transform(y_pred)
        y_test = label_binarizer.inverse_transform(y_test)
        y_train = label_binarizer.inverse_transform(y_train)
        ########################################################################################################################
        # get reports on model performance
        ########################################################################################################################
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))
        print(cm)

        cmPerFold.append(cm)
        report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(Y))
        # record variables per fold
        reports_per_fold.append(report)

        tmp_avg_precision = []
        tmp_avg_recall = []
        tmp_avg_F1 = []
        for cl in np.unique(y_train):
            tmp_avg_precision.append(report[str(cl)]["precision"])
            tmp_avg_recall.append(report[str(cl)]["recall"])
            tmp_avg_F1.append(report[str(cl)]["f1-score"])

        avg_precision = np.mean(tmp_avg_precision)
        avg_recall = np.mean(tmp_avg_recall)
        avg_F1 = np.mean(tmp_avg_F1)

        # print mean values
        avgAccuracy = []
        avgRecall = []
        avgPrecision = []
        avgF1 = []
        for rep in reports_per_fold:
            print(pd.DataFrame(rep))
            avgAccuracy.append(rep["accuracy"])
            for i in np.unique(y_train):
                avgRecall.append(rep[str(i)]["recall"])
                avgPrecision.append(rep[str(i)]["precision"])
                avgF1.append(rep[str(i)]["f1-score"])

        print(
            "Average accuracy: %.3f\nAverage precision:  %.3f\nAverage recall:  %.3f\nAverage F1 : %.3f\n " % (
                np.mean(avgAccuracy), np.mean(avgPrecision), np.mean(avgRecall), np.mean(avgF1)))

        end_time = time.time()
        print("Time elapsed ", str(end_time - start_time))
        reports_df.loc[len(reports_df)] = [OPT_WIN, OPT_OVR, EPOCHS, LR, BATCH_SIZE, DROPOUT,
                                           report["accuracy"],
                                           avg_precision,
                                           avg_recall, avg_F1, ifold, np.mean(avgAccuracy),
                                           np.mean(avgPrecision),
                                           np.mean(avgRecall), np.mean(avgF1), end_time - start_time]
        reports_df.infer_objects().dtypes
        reports_df.to_csv(report_filename, index=False)

        # Reset TensorFlow default graph (for TF1 compatibility)
        if hasattr(tf.compat.v1, 'reset_default_graph'):
            tf.compat.v1.reset_default_graph()

        gc.disable()

    # plot confusion matrix
    # calculate and plot average confusion matrix
    filename = 'avg_cm_5_folds_exp_' + str(expert) + "_" + str(mode)
    utilities.plot_avg_cm(cmPerFold, path_variables.FIGURES_PATH, filename, classNames=np.unique(y_train))

if __name__ == "__main__":
    if len(sys.argv) != 3:  # Expecting two arguments: expert and mode
        print("Usage: python model_eval_cv.py <expert> <mode>")
        sys.exit(1)

    expert_arg = int(sys.argv[1])  # Convert the expert argument to an integer
    mode_arg = sys.argv[2]  # Mode argument remains a string
    if not utilities.all_results_model_eval_exist(expert_arg, mode_arg):
        print(
            f"Running 5-fold cv for expert {expert_arg} for {mode_arg} on model with\noptimal window {config.OPT_WIN}\noptimal overlap {config.OPT_OVR}")
        model_eval_cv(expert_arg, mode_arg)
        print(f"5-fold cv model evaluation for expert {expert_arg} for {mode_arg} is complete. Results are saved.")
    else:
        print(f"All results of 5-fold cv model evaluation for Expert {expert_arg} already exist. Skipping search.")
