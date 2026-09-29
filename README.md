# Iris Flower Classification

A beginner-friendly machine-learning project that compares five classifiers and predicts an Iris species from four flower measurements. It loads the classic Iris dataset directly from scikit-learn; no download, CSV, external API, or network access is needed to train.

## Objective

Predict one of three species from sepal length, sepal width, petal length, and petal width:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

## Dataset

The 150-row Iris dataset is bundled with scikit-learn. `src/train_model.py` calls `load_iris()`, turns its measurements into a pandas DataFrame, and converts the numeric target into readable species names. A `data/iris.csv` file is not required, so the `data/` folder is intentionally omitted.

## Features and target

The four numeric measurement columns are the features (`X`); the species name is the target (`y`). The training script prints dataset shape, columns, types, missing-value counts, descriptive statistics, and counts by species.

## Technologies

Python 3, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit, and ipykernel for running the notebook in VS Code. NumPy is included as a required project dependency and is used by scikit-learn internally.

## Project structure

```text
iris-flower-classification/
├── assets/
│   └── iris_background.jpg          # Local Iris photo used by the app
├── models/                         # Created by training; saved model bundle
├── notebooks/
│   └── iris_analysis.ipynb         # Guided exploration and model comparison
├── src/
│   ├── __init__.py
│   ├── predict.py                  # Terminal prediction interface
│   └── train_model.py              # Exploration, plots, training, evaluation
├── visualizations/                 # Created by training; saved plots
├── app.py                          # Streamlit prediction interface
├── requirements.txt
├── README.md
└── .gitignore
```

`models/` and `visualizations/` are generated when training runs. The model and image files are ignored by Git and can be recreated at any time.

## Windows and VS Code setup

1. Install Python 3 from [python.org](https://www.python.org/downloads/) and select the option to add Python to PATH. Install VS Code and its Microsoft Python extension.
2. In File Explorer, create a folder named `iris-flower-classification`. In VS Code choose **File > Open Folder...** and open it.
3. Open **Terminal > New Terminal**. In PowerShell, create and activate a virtual environment:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt instead and run `.venv\Scripts\activate.bat`, or allow scripts for the current PowerShell user with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
4. Install the requirements:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select the interpreter inside this project's `.venv` folder. For the notebook, choose the same `.venv` as its kernel. The terminal prompt should show `(.venv)` after activation.

## Install dependencies

`requirements.txt` lists the runtime libraries. With the project virtual environment activated, run `python -m pip install -r requirements.txt` from the project root. Using `python -m pip` helps ensure packages are installed for the selected interpreter.

## Train and compare models

From the VS Code terminal at the project root:

```powershell
python src/train_model.py
```

The script makes an 80/20 stratified train/test split with `random_state=42`, which preserves class proportions and makes the split reproducible. Logistic Regression, K-Nearest Neighbors, and Support Vector Machine use `StandardScaler` because their calculations depend on feature scale. Decision Tree and Random Forest do not require scaling. Each model is fitted, predicts the held-out test rows, and prints accuracy and a classification report. The report shows precision, recall, F1-score, and sample counts for each class.

The script prints a comparison table and selects the best test accuracy (using macro F1 to break ties). It then prints the selected model's accuracy, macro precision, macro recall, and macro F1-score. These are calculated each run; no result is hard-coded. The holdout set is used both for comparison and selection, so treat the displayed scores as an educational estimate rather than an unbiased final benchmark.

The fitted model is saved with joblib to `models/iris_model.pkl`. This file is a bundle containing the model, selected model name, and feature/class names. Later code loads it with `joblib.load(...)` and uses the bundled estimator's `predict()` or `predict_proba()` methods.

## Visualizations

Training saves plots into `visualizations/`:

- `species_distribution.png`: number of samples in each class; the built-in dataset is balanced.
- `pairplot.png`: pairwise feature relationships, colored by species; petal measurements tend to separate species clearly.
- `sepal_measurements.png`: sepal length against sepal width, colored by species.
- `petal_measurements.png`: petal length against petal width, colored by species.
- `confusion_matrix.png`: actual classes by row and predicted classes by column for the selected classifier. The diagonal contains correct predictions; off-diagonal cells show errors.

## Make a terminal prediction

Train the model first, then run:

```powershell
python src/predict.py
```

Enter four positive measurements when prompted. For example, `5.9`, `3.0`, `5.1`, and `1.8`. The program prints its predicted species and, when available, the probability for each class. Invalid numbers and a missing model produce a clear message.

## Run the Streamlit app

After training, run:

```powershell
streamlit run app.py
```

Streamlit opens a browser automatically. If it does not, open **http://localhost:8501**. Enter the four measurements and click **Predict Species**. The page displays the predicted species and class probabilities when the model supports them.

The app uses the local `assets/iris_background.jpg` file, so its background does not need an internet connection at runtime. The photo is by Danielle Langlois via Wikimedia Commons and is licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); the app displays the attribution.

## Notebook

Open `notebooks/iris_analysis.ipynb` in VS Code and choose the `.venv` interpreter as its kernel. Run the cells from top to bottom for a guided view of loading and exploring the data, visualizing it, splitting it, comparing models, and drawing a confusion matrix. The training script remains the canonical way to save the model bundle used by the app and terminal predictor.

## Example prediction

For measurements `5.9, 3.0, 5.1, 1.8`, the program returns one of the three species labels and model-generated probabilities. The predicted result and probabilities depend on the model selected by the actual training run.

## Results

The comparison table, classification reports, selected model, and holdout metrics are printed by `src/train_model.py` each time it runs. Results are calculated from the current code and split; this README intentionally does not promise a fixed accuracy.

## Troubleshooting

- **`ModuleNotFoundError`**: Activate `.venv` and run `python -m pip install -r requirements.txt`. Confirm VS Code selected `.venv` with **Python: Select Interpreter**.
- **`python` is not recognized**: Install Python and reopen VS Code; on Windows try `py` in place of `python`, for example `py -3 -m venv .venv`.
- **`pip` is not recognized**: Run `python -m pip install -r requirements.txt` (or `py -m pip ...`) rather than calling `pip` directly.
- **`FileNotFoundError` or model file not found**: Ensure the terminal is in the project root and run `python src/train_model.py` before prediction or the app. Training creates the model and plot directories.
- **`streamlit` command not found**: Activate `.venv`, install requirements, then run `python -m streamlit run app.py`.
- **Wrong Python interpreter**: Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select `.venv\Scripts\python.exe`; open a new terminal afterward.
- **PowerShell refuses virtual-environment activation**: Use Command Prompt and run `.venv\Scripts\activate.bat`, or use the PowerShell execution-policy command in setup.

## Future improvements

Add cross-validation and hyperparameter tuning, keep a separate final test set, add automated tests, and track experiment results for more rigorous comparisons.

## Conclusion

This project demonstrates the complete beginner workflow: loading and exploring data, preparing features, comparing classifiers, evaluating predictions, saving a reusable model, and using it from a terminal or web interface.