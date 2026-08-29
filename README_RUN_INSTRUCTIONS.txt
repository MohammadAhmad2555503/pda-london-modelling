CS5812 Predictive Data Analysis
README RUN INSTRUCTIONS
Student: Mohammad Ahmad Author
Student ID: STUDENT_ID

============================================================
1. PURPOSE OF THIS FILE
============================================================S

This README explains how to run the submitted code for the CS5812 Predictive Data Analysis coursework.

The workflow contains two modelling parts:

1. R workflow:
   - data checking
   - data preparation
   - EDA plots
   - k-means exploratory clustering
   - Decision Tree Regression
   - median baseline model
   - export of train/test files for Python

2. Python workflow:
   - MLP deep learning regression model
   - preprocessing using StandardScaler and OneHotEncoder
   - model evaluation
   - MLP plots and output files

The R file must be run first because it creates the train/test CSV files used by the Python MLP model.

============================================================
2. EXPECTED ZIP FILE STRUCTURE
============================================================

The code archive should be named:

CS5812_STUDENT_ID_Code.zip

After extracting the ZIP, the folder should contain:

CS5812_STUDENT_ID_Code/
│
├── final_london_modelling.csv
├── PDA_R_DecisionTree_Author.Rmd
├── PDA_Python_MLP_Author.py
├── README_RUN_INSTRUCTIONS.txt
│
├── outputs/
│
└── eda_plots/

The outputs/ and eda_plots/ folders may already contain submitted output files. If the code is rerun, new output files will be written into these folders.

============================================================
3. REQUIRED SOFTWARE
============================================================

R requirements:
- R
- RStudio is recommended
- R packages:
  - ggplot2
  - rpart

Python requirements:
- Python 3.10, 3.11, 3.12 or 3.13
- VS Code or any Python editor
- Python packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - tensorflow

============================================================
4. HOW TO RUN THE R WORKFLOW
============================================================

Step 1:
Open the project folder in RStudio.

Step 2:
Open the file:

PDA_R_DecisionTree_Author.Rmd

Step 3:
If required, install the R packages by running:

install.packages("ggplot2")
install.packages("rpart")

Step 4:
Run or knit the full R Markdown file.

The R workflow will:

- load final_london_modelling.csv
- remove duplicate transaction IDs where present
- remove unused columns such as Manhattan distance and technical identifiers
- create log_price
- create sale_year and sale_month
- check that sale_year is correctly parsed as a real year
- create EDA plots
- run k-means exploratory clustering
- train and evaluate the Decision Tree Regression model
- create a median baseline model
- save metrics and predictions
- export train_data_for_python.csv and test_data_for_python.csv for Python

Important expected date check:

The R output should show:

Sale year range : 2018 to 2025
Sale month range: 1 to 12

If sale_year appears as 1 to 31, the date parsing is wrong and the workflow should not be submitted.

Step 5:
After the R file completes, check that these files exist:

outputs/train_data_for_python.csv
outputs/test_data_for_python.csv
outputs/r_decision_tree_metrics.csv
outputs/r_baseline_metrics.csv
outputs/r_model_baseline_comparison.csv
outputs/r_decision_tree_importance.csv
eda_plots/r_decision_tree_actual_vs_predicted.png
eda_plots/r_decision_tree_residuals.png
eda_plots/r_decision_tree_importance.png

============================================================
5. HOW TO RUN THE PYTHON WORKFLOW
============================================================

The Python script should be run after the R workflow has completed.

The Python script uses:

outputs/train_data_for_python.csv
outputs/test_data_for_python.csv

These files are produced by the R workflow. This keeps the Decision Tree and MLP comparison fair because both models use the same train/test split.

Step 1:
Open PowerShell in the extracted project folder.

For example:

cd path\to\CS5812_STUDENT_ID_Code

Step 2:
Create a virtual environment:

py -3.13 -m venv .venv

If py -3.13 does not work, use:

python -m venv .venv

Step 3:
Allow activation in the current PowerShell session:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

Step 4:
Activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Step 5:
Upgrade pip:

python -m pip install --upgrade pip

Step 6:
Install the required Python packages:

pip install pandas numpy matplotlib scikit-learn tensorflow

Step 7:
Run the Python script:

python .\PDA_Python_MLP_Author.py

============================================================
6. EXPECTED PYTHON OUTPUTS
============================================================

The Python script will:

- read the R-exported train/test files
- check that sale_year is a real year
- preprocess numeric and categorical variables
- train the MLP model
- evaluate the model on log(price)
- back-transform predictions using exp()
- evaluate predictions on the original pound-price scale
- save metrics, predictions, plots and model summary

Expected output files include:

outputs/python_mlp_metrics.csv
outputs/python_mlp_predictions.csv
outputs/python_mlp_test_predictions.csv
outputs/python_mlp_summary.txt

Expected plot files include:

eda_plots/python_mlp_training_loss.png
eda_plots/python_mlp_actual_vs_predicted.png
eda_plots/python_mlp_residuals.png

============================================================
7. IMPORTANT RUNNING NOTES
============================================================

1. Always run the R Markdown file first.
   The Python script depends on the train/test files exported by R.

2. Do not run the Python file using a different Python installation if the virtual environment has already been created.
   Activate .venv first, then run the script.

3. Before running the Python MLP script, confirm from the R output that sale_year is a real calendar year range, for example 2018–2025, and sale_month is 1–12. 
If sale_year appears as values such as 1–31, the R date-parsing step has not run correctly; rerun the corrected R Markdown workflow before running Python, 
because the Python model uses the train/test files exported from R.

4. The dataset file must be called:

final_london_modelling.csv

5. The dataset must be in the same folder as the R and Python code.

6. The folders outputs/ and eda_plots/ will be created automatically if they do not already exist.

============================================================
8. OUTPUT FOLDERS
============================================================

The folder outputs/ contains CSV and text outputs, including:

- missing-value summary
- Decision Tree metrics
- baseline metrics
- Decision Tree predictions
- variable importance
- train/test files for Python
- MLP metrics
- MLP predictions
- MLP model summary

The folder eda_plots/ contains PNG figures, including:

- EDA plots
- correlation heatmap
- k-means plots
- Decision Tree diagnostic plots
- MLP diagnostic plots

============================================================
9. REPRODUCIBILITY
============================================================

The random seed is set to 42 in both R and Python.

The intended run order is:

1. Run PDA_R_DecisionTree_Author.Rmd
2. Confirm sale_year is 2018 to 2025
3. Run PDA_Python_MLP_Author.py
4. Check outputs/ and eda_plots/

This reproduces the individual Decision Tree and MLP workflows used in the report.

============================================================
10. SUBMISSION FILES
============================================================

The final submission should include:

1. PDF report:
   CS5812_STUDENT_ID.pdf

2. Code archive:
   CS5812_STUDENT_ID_Code.zip


