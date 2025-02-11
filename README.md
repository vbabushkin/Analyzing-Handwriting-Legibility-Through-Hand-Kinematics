# Analyzing Handwriting Legibility Through Hand Kinematics

**Vahan Babushkin**<sup>1,3</sup>, **Haneen Alsuradi**<sup>1</sup>, **Muhamed Osman Al-Khalil**<sup>2</sup> and  
**Mohamad Eid**<sup>1*</sup>

<sup>1</sup> *Applied Interactive Multimedia Lab, Engineering Division, New York University Abu Dhabi, United Arab Emirates*  
<sup>2</sup> *Arabic Studies program, New York University Abu Dhabi, United Arab Emirates*  
<sup>3</sup> *Tandon School of Engineering, New York University, New York, NY, USA*  

**Correspondence<sup>*</sup>:**  
Mohamad Eid  
[mohamad.eid@nyu.edu](mailto:mohamad.eid@nyu.edu)

## Abstract
**Introduction:** Handwriting is a complex skill that requires coordination between human motor system, sensory perception, cognitive processing, memory retrieval, and linguistic proficiency. Various aspects of hand and stylus kinematics can affect the legibility of a handwritten text. Assessing handwriting legibility is challenging due to variations in experts' cultural and academic backgrounds, which introduce subjectivity biases in evaluations.
	
**Methods:** In this paper, we utilize a deep-learning model to analyze kinematic features influencing the legibility of handwriting based on temporal convolutional networks (TCN). Fifty subjects are recruited to complete a 26-word paragraph handwriting task, designed to include all possible orthographic combinations of Arabic characters, during which the hand and stylus movements are recorded. A total of 117 different spatiotemporal features are recorded, and the data collected are used to train the model. Shapley values are used to determine the important hand and stylus kinematics features towards evaluating legibility. Three experts are recruited to label the produced text into different legibility scores. Statistical analysis of the top 6 features is conducted to investigate the differences between features associated with high and low legibility scores. 
	
**Results:** Although the model trained on stylus kinematics features demonstrates relatively high accuracy (around 76\%), where the number of legibility classes can vary between 7 and 8 depending on the expert, the addition of hand kinematics features significantly increases the model accuracy by approximately 10\%. Explainability analysis revealed that pressure variability, pen slant (altitude, azimuth), and hand speed components are the most prominent for evaluating legibility across the three experts. 
	
**Discussion:** The model learns meaningful stylus and hand kinematics features associated with the legibility of handwriting. The hand kinematics features are important for accurate assessment of handwriting legibility. The proposed approach can be used in handwriting learning tools for personalized handwriting skill acquisition as well as for pathology detection and rehabilitation. 
	
## How to run the code

The code can be run in two ways. 

- Interactive mode - run main.py and follow the instructions. It will ask whether the user wants to preprocess the data and then ask to enter the expert number to conduct the parameter search, model evaluation, explainability and statistical analyses. For model evaluation it will ask if the user wants to run it for 7 stylus kinematics features, 110 hand kinematics features or all 117 stylus and hand kinematics features.

- Manual mode - allows to run each module separately from comman line with specifying parameters manually.
  - to preprocess the data:
    ```bash
    python preprocess_data.py
  - to run search for optimal overlap manually (e.g. for expert 3):
    ```bash
    python param_search_ovr.py 3
  - to run search for optimal window manually (e.g. for expert 1):
    ```bash
    python param_search_win.py 1
  - to manually plot results of optimal overlap search (e.g. for expert 2):
    ```bash
    python plot_param_search_results_ovr.py 2
  - to manually plot results of optimal window search (e.g. for expert 3):
    ```bash
    python plot_param_search_results_win.py 3
  - to manually plot results of optimal window search for all experts:
    ```bash
    python plot_param_search_results_win_all.py 
  - to manually evaluate the model (e.g. for expert 1 with stylus features. Note there are three modes "all", "hand" and "stylus"):
    ```bash
    python model_eval_cv.py 1 "stylus"
  - to manually calculate Shapley values (e.g. for expert 3):
    ```bash
    python calc_shapley_values.py 3 
  - to manually plot Shapley values (e.g. for expert 2):
    ```bash
    python plot_shapley_values.py 2
  - to conduct statistical analysis for all experts:
    ```bash
    python stat_analysis.py

Note that all results such as .csv and .pickle files are saved in RESULTS folder. Plots are saved in FIGURES folder and models are saved in MODELS folder as .h5 files.
