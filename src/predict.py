"""Collect flower measurements in the terminal and predict the species."""

from pathlib import Path
import math

try:
    import joblib
    import pandas as pd
except ImportError as error:
    raise SystemExit(
        "joblib is missing. Activate your virtual environment and run "
        "'python -m pip install -r requirements.txt'."
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.pkl"
PROMPTS = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width",
]


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit("Model file not found. Run 'python src/train_model.py' first.")

    measurements: list[float] = []
    for prompt in PROMPTS:
        try:
            value = float(input(f"Enter {prompt}: "))
        except ValueError as error:
            raise SystemExit(f"{prompt} must be a number, such as 5.9.") from error
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"{prompt} must be a finite number greater than zero.")
        measurements.append(value)

    try:
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        input_data = pd.DataFrame([measurements], columns=bundle["feature_names"])
        prediction = model.predict(input_data)[0]
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Could not load or use the saved model: {error}") from error

    print(f"\nPredicted Iris Species: {prediction}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        print("Prediction probabilities:")
        for species, probability in zip(model.classes_, probabilities):
            print(f"  {species}: {probability:.1%}")


if __name__ == "__main__":
    main()