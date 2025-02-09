import utilities
import param_search_win
import param_search_ovr
import model_eval_cv
import calc_shapley_values
import config
import plot_distributions
import plot_shapley_values
import plot_param_search_results_win
import plot_param_search_results_ovr
import stat_analysis


def main():
    # Ask if the user wants to preprocess the data
    preprocess_choice = input("Do you want to preprocess the data? (yes/no): ").strip().lower()
    if preprocess_choice == 'yes':
        utilities.preprocess()
        print("Data preprocessing complete.")

    # Ask which expert to run
    expert = input("Enter the expert number to run (e.g., 1, 2, 3): ").strip()
    try:
        expert = int(expert)
        # plot distributions
        plot_distributions.plot_distributions(expert)
    except ValueError:
        print("Invalid input. Please enter a valid expert number.")
        return

    # Ask if the user wants to run param_search_win
    run_ovr_search = input("Do you want to run ovrlap parameter search? (yes/no): ").strip().lower()
    if run_ovr_search == 'yes':
        # Check and run param_search_ovr
        if not utilities.all_results_overlap_search_exist(expert):
            param_search_ovr.param_search_ovr(expert)
            print(f"Search for expert {expert} is complete. Results are saved.")
        else:
            print(f"All results for expert {expert} already exist. Skipping search.")
        # plor parameter search results
        plot_param_search_results_ovr()
        # Ask for the optimal overlap value
        opt_ovr = input("Enter the optimal overlap value: ").strip()
        config.OPT_OVR = opt_ovr
        print(f"Optimal overlap value set to: {config.OPT_OVR}")

    # Ask if the user wants to run param_search_win
    run_win_search = input("Do you want to run window parameter search? (yes/no): ").strip().lower()
    if run_win_search == 'yes':
        if not utilities.all_results_win_search_exist(expert):
            param_search_win.param_search_win(expert)
            print(f"Search for expert {expert} is complete. Results are saved.")
        else:
            print(f"All results for expert {expert} already exist. Skipping search.")
        # plor parameter search results
        plot_param_search_results_win()
        # Ask for the optimal window value
        opt_win = input("Enter the optimal window value: ").strip()
        config.OPT_WIN = opt_win
        print(f"Optimal window value set to: {config.OPT_WIN}")

    # Ask if the user wants to evaluate the model
    cv_choice = input("Do you want to run 5-fold cross validation on model? (yes/no): ").strip().lower()
    if cv_choice == 'yes':
        # ask for which mode to run
        mode = input("Enter the mode to run (e.g., all, hand or stylus): ").strip()
        try:
            mode = str(mode)
        except ValueError:
            print("Invalid input. Please enter a valid mode.")
            return

        if not utilities.all_results_model_eval_exist(expert, mode):
            print(
                f"Running 5-fold cv for expert {expert} for {mode} on model with\noptimal window {config.OPT_WIN}\noptimal overlap {config.OPT_OVR}")
            model_eval_cv.model_eval_cv(expert, mode)
            print(f"5-fold cv model evaluation for expert {expert} for {mode} is complete. Results are saved.")
        else:
            print(f"All results of 5-fold cv model evaluation for Expert {expert} already exist. Skipping search.")

    # Ask if the user wants to calculate shapley values
    shap_choice = input("Do you want to calculate Shapley values? (yes/no): ").strip().lower()
    if shap_choice == 'yes':
        if not utilities.all_results_shap_values_exist(expert):
            if not utilities.all_results_model_eval_exist(expert, mode):
                print(f"The files necessaryfor calculating Shapley values for expert {expert} are missing in RESULTS folder.Please run model_eval_cv {expert}.")
            else:
                print(
                    f"Calculating Shapley values for expert {expert} on model with\noptimal window {config.OPT_WIN}\noptimal overlap {config.OPT_OVR}")
                calc_shapley_values.calc_shapley_values(expert)
                # plot Shapley values
                plot_shapley_values.plot_shapley_values(expert)
                print(f"Calculating Shapley values for expert {expert} is complete. Results are saved.")
        else:
            print(f"Shapley values for expert {expert} are already calculated. Skipping search.")

    # Ask if the user wants to conduct statistical analysis
    stat_choice = input("Do you want to perform statistical analysis? (yes/no): ").strip().lower()
    if stat_choice == 'yes':
        # conduct statistical analysis
        stat_analysis()


if __name__ == '__main__':
    main()
