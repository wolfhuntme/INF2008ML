# INF2008ML PROJECT

## Dataset Information
- Training, testing, and development datasets:
  - `signatures_cedar/full_forg`
  - `signatures_cedar/full_org`
- Unseen testing datasets:
  - `signatures_cedar/unseen_data_for_testing/unseen_forg`
  - `signatures_cedar/unseen_data_for_testing/unseen_org`

---

## How to Run

### Step 1: Ensure All Libraries Are Installed
The following libraries are required:

1. numpy: `pip install numpy`
2. OpenCV: `pip install opencv-python`
3. matplotlib: `pip install matplotlib`
4. joblib: `pip install joblib`
5. scikit-image: `pip install scikit-image`
6. scikit-learn: `pip install scikit-learn`
7. seaborn: `pip install seaborn`
8. tabulate: `pip install tabulate`
9. memory_profiler: `pip install memory_profiler`

---

### Step 2: Save Individual Models
Run the following scripts to save each model:
- `saveadaboostmodel.py`
- `SVM.py`
- `random_forest.py`
- `logistic_regression.py`
- `knn.py`

---

### Step 3: Run Ensemble Method
Execute the script to generate individual model probabilities and ensemble method probability:
`python newensemble.py`

---

### Step 4: Evaluate Model Performance
To evaluate the performance metrics of the models, run:
`python testModelEvaluation.py`
