"""Explore Iris, compare classifiers, and save the best model and plots."""

from pathlib import Path

try:
    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
except ImportError as error:
    raise SystemExit(
        "A required package is missing. Activate your virtual environment and run "
        "'python -m pip install -r requirements.txt'."
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.pkl"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_NAMES = ["sepal length", "sepal width", "petal length", "petal width"]


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Load sklearn's built-in data and return named features and species."""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    class_names = [f"Iris-{name}" for name in iris.target_names]
    data["species"] = [class_names[target] for target in iris.target]
    return data, class_names


def explore_dataset(data: pd.DataFrame) -> None:
    """Print a compact overview before model training."""
    print("\n--- First five rows ---")
    print(data.head())
    print(f"\nDataset shape: {data.shape}")
    print(f"\nColumns: {data.columns.tolist()}")
    print("\nData types:")
    print(data.dtypes)
    print("\nMissing values per column:")
    print(data.isnull().sum())
    print("\nFeature summary:")
    print(data[FEATURE_NAMES].describe())
    print("\nSamples per species:")
    print(data["species"].value_counts().sort_index())


def save_data_visualizations(data: pd.DataFrame) -> None:
    """Create and save the requested exploratory plots."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="species", hue="species", legend=False)
    plt.title("Iris Species Distribution")
    plt.xlabel("Species")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "species_distribution.png", dpi=150)
    plt.close()

    pair_grid = sns.pairplot(data, hue="species", corner=True, diag_kind="hist")
    pair_grid.figure.suptitle("Iris Feature Pair Plot", y=1.02)
    pair_grid.savefig(VISUALIZATIONS_DIR / "pairplot.png", dpi=150)
    plt.close(pair_grid.figure)

    for x_feature, y_feature, filename, title in [
        ("sepal length", "sepal width", "sepal_measurements.png", "Sepal Length vs Sepal Width"),
        ("petal length", "petal width", "petal_measurements.png", "Petal Length vs Petal Width"),
    ]:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=data,
            x=x_feature,
            y=y_feature,
            hue="species",
            s=65,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / filename, dpi=150)
        plt.close()


def build_models() -> dict[str, object]:
    """Return classifiers with scaling where distance/margins require it."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", CalibratedClassifierCV(SVC(), cv=5)),
            ]
        ),
    }


def main() -> None:
    try:
        data, class_names = load_dataset()
        explore_dataset(data)
        save_data_visualizations(data)

        features = data[FEATURE_NAMES]
        target = data["species"]
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )

        models = build_models()
        results: list[dict[str, object]] = []
        fitted_models: dict[str, object] = {}
        print(f"\nTraining on {len(x_train)} rows; testing on {len(x_test)} rows.")

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            macro_f1 = f1_score(y_test, predictions, average="macro")
            results.append({"Model": model_name, "Accuracy": accuracy, "Macro F1": macro_f1})
            fitted_models[model_name] = model
            print(f"\n--- {model_name} ---")
            print(f"Accuracy: {accuracy:.2%}")
            print(classification_report(y_test, predictions, labels=class_names, zero_division=0))

        results.sort(key=lambda result: (result["Accuracy"], result["Macro F1"]), reverse=True)
        print("\n--- Model comparison ---")
        print(
            pd.DataFrame(results)
            .assign(Accuracy=lambda table: table["Accuracy"].map(lambda value: f"{value:.2%}"))
            .drop(columns="Macro F1")
            .to_string(index=False)
        )

        best_result = results[0]
        best_name = str(best_result["Model"])
        best_model = fitted_models[best_name]
        best_predictions = best_model.predict(x_test)
        matrix = confusion_matrix(y_test, best_predictions, labels=class_names)
        print(f"\nSelected model: {best_name}")
        print(f"Accuracy: {accuracy_score(y_test, best_predictions):.2%}")
        print(f"Macro precision: {precision_score(y_test, best_predictions, average='macro', zero_division=0):.3f}")
        print(f"Macro recall: {recall_score(y_test, best_predictions, average='macro', zero_division=0):.3f}")
        print(f"Macro F1-score: {f1_score(y_test, best_predictions, average='macro', zero_division=0):.3f}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
        )
        plt.title(f"Confusion Matrix: {best_name}")
        plt.xlabel("Predicted species")
        plt.ylabel("Actual species")
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / "confusion_matrix.png", dpi=150)
        plt.close()

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": best_model,
                "model_name": best_name,
                "feature_names": FEATURE_NAMES,
                "class_names": class_names,
            },
            MODEL_PATH,
        )
        print(f"\nSaved model bundle to: {MODEL_PATH}")
        print(f"Saved visualizations to: {VISUALIZATIONS_DIR}")
    except OSError as error:
        raise SystemExit(f"Could not read or write a project file: {error}") from error


if __name__ == "__main__":
    main()