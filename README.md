# Dynamic Feature Selection

We use Monte Carlo REINFORCE to design an algorithm to not only find the optimal deep learning architecture but also the optimal set of features that can maximize the performance of the said deep leaning model. We take up the problem of predicting the onset of severe sepsis before a 4-hour period for validating our methods and compare them with existing severe sepsis literature. Sepsis is a life-threatening condition caused by the patient body’s extreme response to an infection, causing tissue damage and multiple organ failures. We use MIMIC-III dataset, a publicly available medical dataset for our experiments. Apart from the 6 common vital sign measurements, the dataset also contains 127 physiological and laboratory features to predict the onset of severe sepsis, mostly observed in intensive care units (ICUs), which match the cohort of this study: the MIMIC-III dataset for patients admitted in ICUs. We aim to use reinforcement learning to reduce the number of features (from 133) without sacrificing peak model performance that uses all 133 features. Among the discovered deep learning models, the CNN-LSTM model using 110 features achieves the best performance: an AUC of 0.933 in predicting the onset of severe sepsis.

## Code
This project uses reinforcement learning and deep learning built using pytorch. The project is well structured into different classes. the class files are located in the src folder. The pytorch model is saved in the model folder.

## Install dependencies
Provided in the requirements file

## Running the code
python dynamic_selection.py

Dataset is not provided due to privacy restrictions. Running the code requires the appropriate data for sepsis (your require permission from MIMIC-III Clinical Database PhysioNet).
