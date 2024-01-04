import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_dataset(category="house"):
    # param `category` should be one of ["house", "wine", "news"]
    if category == "house":
        df = pd.read_csv("../data/HousingData.csv")

        # Fill the NaNs
        df = df.fillna(df.median())

        # Normalize all continuous features
        scaler = StandardScaler()
        df.loc[:, df.columns != "MEDV"] = scaler.fit_transform(
            df.loc[:, df.columns != "MEDV"]
        )

        return df.loc[:, df.columns != "MEDV"], df["MEDV"]

    elif category == "wine":
        from ucimlrepo import fetch_ucirepo

        base = fetch_ucirepo(id=186)
        return base.data.features, base.data.targets

    elif category == "news":
        df = pd.read_csv("../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
        df.rename(columns=lambda x: x.strip(), inplace=True)
        df = df.drop(["url", "timedelta"], axis=1)
        df = df.fillna(df.mean())

        # Remove outliers of "shares"
        sorted_share = df.sort_values(by="shares")["shares"]
        Q1 = sorted_share.quantile(0.25)
        Q3 = sorted_share.quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df["shares"] >= Q1 - 1.5 * IQR) & (df["shares"] <= Q3 + 1.5 * IQR)]
        print(f"Remove {len(sorted_share) - len(df)} outliers")

        # I give up!
        # Just do it as a classification task
        # mean = df["shares"].mean()
        # df["shares"] = df["shares"].apply(lambda x: 1 if x > mean else 0)

        # Scaling and SMOTE is needed because data is not properly distributed
        X = df.loc[:, df.columns != "shares"]
        y = df["shares"]
        scaler = StandardScaler()
        # smote = SMOTE()
        X = scaler.fit_transform(X)
        # X, y = smote.fit_resample(X, y)

        return pd.DataFrame(X), pd.DataFrame(y)

    else:
        raise ValueError


if __name__ == "__main__":
    pass
