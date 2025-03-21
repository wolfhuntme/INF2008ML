# INF2008ML PROJECT
Dataset used for training, test and development: signatures_cedar/full_forg and signatures_cedar/full_org
Data used for unseen testing: signatures_cedar/unseen_data_for_testing/unseen_forg and signatures_cedar/unseen_data_for_testing/unseen_org

How to run:
1: Ensure all libraries are installed.
    Library used:
        1. numpy: pip install numpy
        2. cv2: pip install opencv-python
        3. matplotlib: pip install matplotlib
        4. joblib: pip install joblib
        5. scikit-image: pip install scikit-image
        6. scikit-learn: pip install scikit-learn
        7. seaborn: pip install seaborn
        8. tabulate: pip install tabulate
        9. memory_profiler: pip install memory_profiler
2: Run the individual models to save their model. (saveadaboostmodel.py, SVM.py, random_forest.py, logistic_regression.py, knn.py)
3: Run newensemble.py to for the individual model's probability and ensembled method probability
4: Run testModelEvaluation.py for the model's performance metrics
