"""
Real Estate Price Prediction - model training pipeline
(converted from notebooks/real_estate_price_model.ipynb, bugs fixed)

Fixes vs. the original notebook:
- Reload cell opened `pickle.load(open('RE_Model','rb'))` but the model was saved
  as `Re_Model.pkl` -> FileNotFoundError. Now one MODEL_PATH constant is used
  everywhere (renamed to real_estate_model.pkl to match the app/README).
- `mean_absolute_error(pred, y)` argument order flipped to (y_true, y_pred).
- Final sanity prediction used a bare list (no feature names, unlabeled magic
  numbers) -> now a DataFrame row with named columns.
- Duplicate train/test-split cell removed; random_state added everywhere so
  results are reproducible.
- Data/model paths resolved relative to this file.

Trains Linear Regression, Decision Tree, and Random Forest; compares MAE;
saves the Random Forest as the deployed model.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # allow running headless
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "final.csv")
MODEL_PATH = os.path.join(HERE, "models", "real_estate_model.pkl")
TREE_PLOT_PATH = os.path.join(HERE, "assets", "tree.png")
RANDOM_STATE = 567


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("price", axis=1)
    y = df["price"]
    return X, y


def evaluate(name, model, X_train, y_train, X_test, y_test):
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"{name:<18} train MAE: {train_mae:>12,.2f}   test MAE: {test_mae:>12,.2f}")
    return test_mae


def main():
    X, y = load_data()

    # keep the Condo/House mix identical in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=X["property_type_Condo"],
        random_state=RANDOM_STATE,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    # 1) Linear Regression (baseline)
    lr = LinearRegression().fit(X_train, y_train)
    evaluate("LinearRegression", lr, X_train, y_train, X_test, y_test)

    # 2) Decision Tree
    dt = DecisionTreeRegressor(max_depth=3, max_features=10,
                               random_state=RANDOM_STATE).fit(X_train, y_train)
    evaluate("DecisionTree", dt, X_train, y_train, X_test, y_test)

    plt.figure(figsize=(20, 10))
    tree.plot_tree(dt, feature_names=list(dt.feature_names_in_), filled=True)
    plt.savefig(TREE_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Decision tree plot saved to {TREE_PLOT_PATH}")

    # 3) Random Forest (deployed model)
    rf = RandomForestRegressor(n_estimators=200, criterion="absolute_error",
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate("RandomForest", rf, X_train, y_train, X_test, y_test)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"\nModel saved to {MODEL_PATH}")

    # reload and sanity-check with a named, labeled example row
    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)

    example = pd.DataFrame([{
        "year_sold": 2012, "property_tax": 216, "insurance": 74,
        "beds": 1, "baths": 1, "sqft": 618, "year_built": 2000,
        "lot_size": 600, "basement": 1, "popular": 0, "recession": 0,
        "property_age": 12, "property_type_Condo": 1,
    }])[list(loaded.feature_names_in_)]
    print(f"Sanity-check prediction: {loaded.predict(example)[0]:,.2f}")


if __name__ == "__main__":
    main()
# end of file
