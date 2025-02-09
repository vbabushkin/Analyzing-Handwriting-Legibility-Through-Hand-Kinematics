import numpy as np
import pandas as pd
import config
import path_variables
import utilities

# constants
RESULTS_PATH = path_variables.RESULTS_PATH
FIGURES_PATH = path_variables.FIGURES_PATH
DATA_PATH = path_variables.MAIN_DATA_FOLDER
winSize = config.OPT_WIN
ovr = config.OPT_OVR
overlap = int(winSize * ovr)

########################################################################################################################
# get train and test sets split into windows
########################################################################################################################
def preprocess_for_stat_analysis():
    dfExpertLabels = pd.read_csv(DATA_PATH + "avg_legibility_score.csv")
    (X, Y, subjInfo) = utilities.load_data()
    Y = np.array(Y)

    exp_subj_info = np.zeros((subjInfo.shape[0], 5))
    for exp_idx in [2, 3, 4]:
        for k in range(1, 51):
            exp_subj_info[np.where(subjInfo[:, 0] == k)[0], 0:2] = subjInfo[np.where(subjInfo[:, 0] == k)[0], 0:2]
            exp_subj_info[np.where(subjInfo[:, 0] == k)[0], exp_idx] = np.mean(
                subjInfo[np.where(subjInfo[:, 0] == k)[0], exp_idx])

    exp_subj_info[:, 2] = np.round(exp_subj_info[:, 2]).astype(int)
    exp_subj_info[:, 3] = np.round(exp_subj_info[:, 3]).astype(int)
    exp_subj_info[:, 4] = np.round(exp_subj_info[:, 4]).astype(int)

    subjInfo = exp_subj_info

    labels_exp_1 = subjInfo[:, 2]
    labels_exp_2 = subjInfo[:, 3]
    labels_exp_3 = subjInfo[:, 4]

    min_exp_1 = min(labels_exp_1)
    min_exp_2 = min(labels_exp_2)
    min_exp_3 = min(labels_exp_3)

    max_exp_1 = max(labels_exp_1)
    max_exp_2 = max(labels_exp_2)
    max_exp_3 = max(labels_exp_3)

    # get high/low legibility labels
    high_legibility_idx = []
    low_legibility_idx = []
    for i in range(subjInfo.shape[0]):
        l_exp_1 = np.nan
        l_exp_2 = np.nan
        l_exp_3 = np.nan
        h_exp_1 = np.nan
        h_exp_2 = np.nan
        h_exp_3 = np.nan
        if subjInfo[i, 2] == min_exp_1:
            l_exp_1 = i
        if subjInfo[i, 3] == min_exp_2:
            l_exp_2 = i
        if subjInfo[i, 4] == min_exp_3:
            l_exp_3 = i
        low_legibility_idx.append([l_exp_1, l_exp_2, l_exp_3])

        if subjInfo[i, 2] == max_exp_1:
            h_exp_1 = i
        if subjInfo[i, 3] == max_exp_2:
            h_exp_2 = i
        if subjInfo[i, 4] == max_exp_3:
            h_exp_3 = i
        high_legibility_idx.append([h_exp_1, h_exp_2, h_exp_3])

    high_legibility_idx = np.array(high_legibility_idx)
    low_legibility_idx = np.array(low_legibility_idx)

    h_idx = []
    l_idx = []
    for i in range(high_legibility_idx.shape[0]):
        if not all(np.isnan(high_legibility_idx[i, :])):
            h_idx.append(i)
        if not all(np.isnan(low_legibility_idx[i, :])):
            l_idx.append(i)

    # split into windows
    for i in range(len(X)):
        p_idx = np.where(np.array(config.features_117) == "pressure")[0][0]
        azimuth_idx = np.where(np.array(config.features_117) == "azimuth")[0][0]
        altitude_idx = np.where(np.array(config.features_117) == "altitude")[0][0]
        v_x_idx = np.where(np.array(config.features_117) == "handSpeed_x")[0][0]
        v_y_idx = np.where(np.array(config.features_117) == "handSpeed_y")[0][0]
        v_z_idx = np.where(np.array(config.features_117) == "handSpeed_z")[0][0]

        tmpX = X[i]
        # sliding window
        a = utilities.get_strides(tmpX, winSize, overlap)

        azimuth_array = []
        altitude_array = []
        v_array = []
        press_array = []

        for j in range(a.shape[0]):
            px = a[j,:,p_idx]
            press_array.append(np.std(a[j, :, p_idx]))

            azimuth = a[j, :, azimuth_idx]
            altitude = a[j, :, altitude_idx]
            azimuth_array.append(np.mean(azimuth))
            altitude_array.append(np.mean(altitude))

            v_x = a[j, :, v_x_idx]
            v_y = a[j, :, v_y_idx]
            v_z = a[j, :, v_z_idx]
            v = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
            v_array.append(np.mean(v))

        p_var = press_array
        azimuth_var = azimuth_array
        altitude_var = altitude_array
        v_var = v_array

        if i in l_idx:
            press_variab_low = np.hstack((locals().get('press_variab_low', p_var), p_var))
            azimuth_low = np.hstack((locals().get('azimuth_low', azimuth_var), azimuth_var))
            altitude_low = np.hstack((locals().get('altitude_low', altitude_var), altitude_var))
            v_low = np.hstack((locals().get('v_low', v_var), v_var))

        if i in h_idx:
            press_variab_high = np.hstack((locals().get('press_variab_high', p_var), p_var))
            azimuth_high = np.hstack((locals().get('azimuth_high', azimuth_var), azimuth_var))
            altitude_high = np.hstack((locals().get('altitude_high', altitude_var), altitude_var))
            v_high = np.hstack((locals().get('v_high', v_var), v_var))

    # get data for correlations plot (subject-based)
    pressure_var_array = []
    velocity_avg_array = []

    h_idx = np.array(h_idx)
    l_idx = np.array(l_idx)

    h_idx_subj = []
    l_idx_subj = []

    for i in range(1, config.NUM_SUBJECTS+1):
        sub_idx = np.where(subjInfo[:, 0] == i)[0]
        tmp_p_var = []
        tmp_v_avg = []
        for k in sub_idx:
            tmpX = X[k]
            v_x = tmpX[:, v_x_idx]
            v_y = tmpX[:, v_y_idx]
            v_z = tmpX[:, v_z_idx]
            p = tmpX[:, p_idx]
            v = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)

            p_var = np.std(p)
            v_avg = np.mean(v)
            tmp_p_var.append(p_var)
            tmp_v_avg.append(v_avg)

        any_h_idx = np.intersect1d(sub_idx, h_idx)  #
        any_l_idx = np.intersect1d(sub_idx, l_idx)

        if len(any_h_idx) > len(
                any_l_idx):  # if the number of high legibility paragraphs prevail -- subject's handwriting is of high legibility
            h_idx_subj.append(i)
        elif len(any_h_idx) < len(
                any_l_idx):  # if the number of low legibility paragraphs prevail -- subject's handwriting is of low legibility
            l_idx_subj.append(i)
        # if in case number of high legibility paragraphs equal to number of low legibility paragraphs -- subject is "other"-- average legibility

        p_v = np.vstack((tmp_p_var, tmp_v_avg))
        p_v_sorted = np.sort(p_v)

        pressure_var_array.append(np.mean(p_v_sorted[0]))
        velocity_avg_array.append(np.mean(p_v_sorted[1]))

    return (press_variab_low,press_variab_high, azimuth_low, azimuth_high,altitude_low, altitude_high,v_low, v_high, pressure_var_array,velocity_avg_array, l_idx_subj, h_idx_subj)

def stat_analysis():
    (press_variab_low, press_variab_high, azimuth_low, azimuth_high, altitude_low, altitude_high, v_low, v_high, pressure_var_array, velocity_avg_array, l_idx_subj, h_idx_subj) = preprocess_for_stat_analysis()
    #####################################################################################################################
    ### pressure variability
    #####################################################################################################################
    label = "pressure variability"
    feature_name = "press_variability"
    utilities.conduct_stat_tests(feature_name, label, press_variab_low, press_variab_high)

    #####################################################################################################################
    ### azimuth
    #####################################################################################################################
    label = "azimuth"
    feature_name = "azimuth"
    utilities.conduct_stat_tests(feature_name, label, azimuth_low, azimuth_high)

    #####################################################################################################################
    ### altitude
    #####################################################################################################################
    label = "altitude"
    feature_name = "altitude"
    utilities.conduct_stat_tests(feature_name, label, altitude_low, altitude_high)

    #####################################################################################################################
    ### abs. velocity
    #####################################################################################################################
    label = "absolute velocity"
    feature_name = "abs_velocity"
    utilities.conduct_stat_tests(feature_name, label, v_low, v_high)

    #####################################################################################################################
    ### plot correlations between pressure variability and abs. velocity
    #####################################################################################################################
    # normalize
    pressure_var_array_norm = (pressure_var_array - np.min(pressure_var_array)) / (
                np.max(pressure_var_array) - np.min(pressure_var_array))
    velocity_avg_array_norm = (velocity_avg_array - np.min(velocity_avg_array)) / (
                np.max(velocity_avg_array) - np.min(velocity_avg_array))

    utilities.plot_correlations(pressure_var_array_norm, velocity_avg_array_norm, l_idx_subj, h_idx_subj,
                                x_lab="pressure variability", y_lab ="mean absolute velocity",
                                filename ="avg_velocity_pressure_var_low_high_legibility_subjects_abs.pdf")

if __name__ == "__main__":
    print("Usage: python stat_analysis.py")
    stat_analysis()
    print("Finalizing statistical analysis: the plots are stored in FIGURES folder.")