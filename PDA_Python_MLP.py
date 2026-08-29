# CS5812 Predictive Data Analysis
# Deep Learning model: MLP regression
# Mohammad Ahmad Author

from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "outputs"
plot_dir = base_dir / "eda_plots"

output_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)

train_path = output_dir / "train_data_for_python.csv"
test_path = output_dir / "test_data_for_python.csv"
full_data_path = base_dir / "final_london_modelling.csv"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

if train_path.exists() and test_path.exists():
    print("Using train/test files exported from R.")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

else:
    print("R train/test files not found. Creating train/test split in Python.")

    if not full_data_path.exists():
        raise FileNotFoundError("final_london_modelling.csv was not found in the project folder.")

    df = pd.read_csv(full_data_path)

    drop_cols = [
        "X",
        "X.1",
        "manhattan_station_km",
        "dist_to_tube_km",
        "n_exact_keys",
        "locality",
        "postcode",
        "transaction_id",
        "street_split_clean"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "price" not in df.columns:
        raise ValueError("The dataset must contain a price column.")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0]
    df["log_price"] = np.log(df["price"])

    if "date_of_transfer" in df.columns:
        df["date_of_transfer"] = pd.to_datetime(
            df["date_of_transfer"],
            errors="coerce",
            dayfirst=True
        )
        df["sale_year"] = df["date_of_transfer"].dt.year
        df["sale_month"] = df["date_of_transfer"].dt.month

    model_cols = [
        "price",
        "log_price",
        "current_energy_efficiency",
        "potential_energy_efficiency",
        "built_form",
        "total_floor_area",
        "number_habitable_rooms",
        "construction_age_band",
        "floor_level",
        "heating_cost_current",
        "heating_cost_potential",
        "hot_water_cost_current",
        "hot_water_cost_potential",
        "lighting_cost_current",
        "old_new",
        "duration",
        "district",
        "epc_source",
        "epc_timing_flag",
        "lat",
        "long",
        "straight_line_station_km",
        "station_zone",
        "property_type",
        "sale_year",
        "sale_month"
    ]

    model_cols = [c for c in model_cols if c in df.columns]
    df = df[model_cols].dropna()

    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

print("Training rows:", train_data.shape[0])
print("Testing rows:", test_data.shape[0])
print("Training columns:", train_data.columns.tolist())


# ------------------------------------------------------------
# Target and predictors
# ------------------------------------------------------------

target = "log_price"

if target not in train_data.columns:
    if "price" in train_data.columns:
        train_data["price"] = pd.to_numeric(train_data["price"], errors="coerce")
        test_data["price"] = pd.to_numeric(test_data["price"], errors="coerce")

        train_data = train_data[train_data["price"] > 0]
        test_data = test_data[test_data["price"] > 0]

        train_data["log_price"] = np.log(train_data["price"])
        test_data["log_price"] = np.log(test_data["price"])
    else:
        raise ValueError("Neither log_price nor price was found.")

X_train = train_data.drop(columns=[c for c in ["price", "log_price"] if c in train_data.columns])
y_train = train_data[target]

X_test = test_data.drop(columns=[c for c in ["price", "log_price"] if c in test_data.columns])
y_test = test_data[target]


# ------------------------------------------------------------
# Identify numeric and categorical columns
# ------------------------------------------------------------

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# ------------------------------------------------------------
# Missing value handling
# ------------------------------------------------------------

for col in numeric_cols:
    median_value = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

    X_train[col] = X_train[col].replace(["nan", "None", "<NA>"], "Missing")
    X_test[col] = X_test[col].replace(["nan", "None", "<NA>"], "Missing")

print("Missing values handled.")


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_train_processed = X_train_processed.astype("float32")
X_test_processed = X_test_processed.astype("float32")

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# ------------------------------------------------------------
# Build MLP model
# ------------------------------------------------------------

input_dim = X_train_processed.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

predicted_log = model.predict(X_test_processed).flatten()
actual_log = y_test.to_numpy()


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

rmse_log = np.sqrt(mean_squared_error(actual_log, predicted_log))
mae_log = mean_absolute_error(actual_log, predicted_log)
r2_log = r2_score(actual_log, predicted_log)

actual_price = np.exp(actual_log)
predicted_price = np.exp(predicted_log)

rmse_price = np.sqrt(mean_squared_error(actual_price, predicted_price))
mae_price = mean_absolute_error(actual_price, predicted_price)
r2_price = r2_score(actual_price, predicted_price)

print("\nMLP results on log price")
print("RMSE:", round(rmse_log, 4))
print("MAE :", round(mae_log, 4))
print("R2  :", round(r2_log, 4))

print("\nMLP results on original price scale")
print("RMSE:", round(rmse_price, 2))
print("MAE :", round(mae_price, 2))
print("R2  :", round(r2_price, 4))


# ------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------

metrics_df = pd.DataFrame({
    "model": ["MLP"],
    "rmse_log": [rmse_log],
    "mae_log": [mae_log],
    "r2_log": [r2_log],
    "rmse_price": [rmse_price],
    "mae_price": [mae_price],
    "r2_price": [r2_price]
})

metrics_df.to_csv(output_dir / "python_mlp_metrics.csv", index=False)


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

predictions_df = pd.DataFrame({
    "actual_log": actual_log,
    "predicted_log": predicted_log,
    "actual_price": actual_price,
    "predicted_price": predicted_price,
    "residual": actual_log - predicted_log
})

predictions_df.to_csv(output_dir / "python_mlp_predictions.csv", index=False)

test_predictions = test_data.copy()
test_predictions["actual_log"] = actual_log
test_predictions["predicted_log"] = predicted_log
test_predictions["actual_price"] = actual_price
test_predictions["predicted_price"] = predicted_price
test_predictions["residual"] = actual_log - predicted_log

test_predictions.to_csv(output_dir / "python_mlp_test_predictions.csv", index=False)


# ------------------------------------------------------------
# Training loss plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("MLP training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "python_mlp_training_loss.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# Actual vs predicted plot
# ------------------------------------------------------------

plot_sample = predictions_df.sample(
    n=min(10000, len(predictions_df)),
    random_state=42
)

plt.figure(figsize=(8, 5))
plt.scatter(
    plot_sample["actual_log"],
    plot_sample["predicted_log"],
    alpha=0.25
)

min_val = min(plot_sample["actual_log"].min(), plot_sample["predicted_log"].min())
max_val = max(plot_sample["actual_log"].max(), plot_sample["predicted_log"].max())

plt.plot([min_val, max_val], [min_val, max_val])
plt.title("MLP: actual vs predicted log price")
plt.xlabel("Actual log price")
plt.ylabel("Predicted log price")
plt.tight_layout()
plt.savefig(plot_dir / "python_mlp_actual_vs_predicted.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# Residual plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(
    plot_sample["predicted_log"],
    plot_sample["residual"],
    alpha=0.2
)
plt.axhline(y=0)
plt.title("MLP residual plot")
plt.xlabel("Predicted log price")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(plot_dir / "python_mlp_residuals.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# Save model summary
# ------------------------------------------------------------

with open(output_dir / "python_mlp_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
    f.write("\n")
    f.write(f"RMSE_LOG: {rmse_log:.4f}\n")
    f.write(f"MAE_LOG: {mae_log:.4f}\n")
    f.write(f"R2_LOG: {r2_log:.4f}\n")
    f.write(f"RMSE_PRICE: {rmse_price:.2f}\n")
    f.write(f"MAE_PRICE: {mae_price:.2f}\n")
    f.write(f"R2_PRICE: {r2_price:.4f}\n")


print("\nPython MLP completed successfully.")
print("CSV outputs saved in:", output_dir)
print("Plots saved in:", plot_dir)
