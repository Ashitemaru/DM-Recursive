import os
import sys
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from utils import load_dataset
from cart import RegressionTree

CATEGORY = "wine"
MAX_DEPTH = {
    "house": 10,
    "wine": 10,
    "news": 8,
}
USE_HANDCRAFT = True


def main():
    # Load data & preprocess
    X, y = load_dataset(CATEGORY)

    # K-fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model_r2_scores = {
        "RT": [],
        "Ref": [],
        "XGB": [],
    }
    model_mse_scores = {
        "RT": [],
        "Ref": [],
        "XGB": [],
    }
    model_training_r2_scores = {
        "RT": [],
        "Ref": [],
        "XGB": [],
    }
    model_training_mse_scores = {
        "RT": [],
        "Ref": [],
        "XGB": [],
    }
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

        # Create model
        if USE_HANDCRAFT:
            model = RegressionTree(max_depth=MAX_DEPTH[CATEGORY])
        ref_model = DecisionTreeRegressor(max_depth=MAX_DEPTH[CATEGORY])
        xg_reg = xgb.XGBRegressor()

        # Train model
        if USE_HANDCRAFT:
            model.fit(X_train, y_train)
            print(f"Fold {i}: Regression Tree Training Finished")
        ref_model.fit(X_train, y_train)
        print(f"Fold {i}: Ref Regression Tree Training Finished")
        xg_reg.fit(X_train, y_train)
        print(f"Fold {i}: XGBoost Training Finished")

        # Predict & test
        if USE_HANDCRAFT:
            y_predict = model.predict(X_test)
        ref_y_predict = ref_model.predict(X_test)
        xg_reg_y_predict = xg_reg.predict(X_test)

        if USE_HANDCRAFT:
            y_train_predict = model.predict(X_train)
        ref_y_train_predict = ref_model.predict(X_train)
        xg_reg_y_train_predict = xg_reg.predict(X_train)

        # If mode is news, we need to round the result
        if CATEGORY == "news":
            if USE_HANDCRAFT:
                y_predict = y_predict.round()
            ref_y_predict = ref_y_predict.round()
            xg_reg_y_predict = xg_reg_y_predict.round()
            if USE_HANDCRAFT:
                y_train_predict = y_train_predict.round()
            ref_y_train_predict = ref_y_train_predict.round()
            xg_reg_y_train_predict = xg_reg_y_train_predict.round()

        # Append results
        if USE_HANDCRAFT:
            model_r2_scores["RT"].append(r2_score(y_test, y_predict))
        model_r2_scores["Ref"].append(r2_score(y_test, ref_y_predict))
        model_r2_scores["XGB"].append(r2_score(y_test, xg_reg_y_predict))
        if USE_HANDCRAFT:
            model_mse_scores["RT"].append(mean_squared_error(y_test, y_predict))
        model_mse_scores["Ref"].append(mean_squared_error(y_test, ref_y_predict))
        model_mse_scores["XGB"].append(mean_squared_error(y_test, xg_reg_y_predict))
        if USE_HANDCRAFT:
            model_training_r2_scores["RT"].append(r2_score(y_train, y_train_predict))
        model_training_r2_scores["Ref"].append(r2_score(y_train, ref_y_train_predict))
        model_training_r2_scores["XGB"].append(
            r2_score(y_train, xg_reg_y_train_predict)
        )
        if USE_HANDCRAFT:
            model_training_mse_scores["RT"].append(
                mean_squared_error(y_train, y_train_predict)
            )
        model_training_mse_scores["Ref"].append(
            mean_squared_error(y_train, ref_y_train_predict)
        )
        model_training_mse_scores["XGB"].append(
            mean_squared_error(y_train, xg_reg_y_train_predict)
        )

    # Print average
    print("R2 Score")
    for model_name, scores in model_r2_scores.items():
        if not USE_HANDCRAFT and model_name == "RT":
            continue
        print(f"{model_name}: {sum(scores) / len(scores) if len(scores) > 0 else 0}")
    print("MSE Score")
    for model_name, scores in model_mse_scores.items():
        if not USE_HANDCRAFT and model_name == "RT":
            continue
        print(f"{model_name}: {sum(scores) / len(scores) if len(scores) > 0 else 0}")
    print("Training R2 Score")
    for model_name, scores in model_training_r2_scores.items():
        if not USE_HANDCRAFT and model_name == "RT":
            continue
        print(f"{model_name}: {sum(scores) / len(scores) if len(scores) > 0 else 0}")
    print("Training MSE Score")
    for model_name, scores in model_training_mse_scores.items():
        if not USE_HANDCRAFT and model_name == "RT":
            continue
        print(f"{model_name}: {sum(scores) / len(scores) if len(scores) > 0 else 0}")


if __name__ == "__main__":
    main()
