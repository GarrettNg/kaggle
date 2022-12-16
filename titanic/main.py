"""
https://www.kaggle.com/competitions/titanic/
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


OUTPUT_DIR_PATH = "./output"
TEST_DATA_PATH = "./data/test.csv"
TRAINING_DATA_PATH = "./data/train.csv"


def check_input(input_files: list) -> None:
    """Print head of input csv files
    :param input_files: list of input files
    """
    dfs = [pd.read_csv(f) for f in input_files]
    [print(df.head()) for df in dfs]


def get_feature_rate(
    dataset: pd.DataFrame, feature_header: str, feature_value: str, feature_outcome: str
) -> float:
    """Get survival rate for a given feature
    :param dataset: DataFrame containing input data
    :param feature_header: feature header from data set
    :param feature_value: feature value from data set
    :param feature_outcome: feature outcome from data set
    :return: feature rate
    """
    feature = dataset.loc[dataset[feature_header] == feature_value][feature_outcome]
    rate = sum(feature) / len(feature)
    print(
        f"% of '{feature_value}' ('{feature_header}') with outcome of '{feature_outcome}':",
        rate,
    )
    return rate


def train_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir_path: str = OUTPUT_DIR_PATH,
) -> None:
    """Train and predict survival rate
    :param train_data: DataFrame containing training data set
    :param test_data: DataFrame containing test data set
    :param output_dir_path: path to write csv file with predictions
    """
    y = train_data["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {"PassengerId": test_data.PassengerId, "Survived": predictions}
    )
    output.to_csv(output_dir_path + "/submission.csv", index=False)


if __name__ == "__main__":
    # check_input([TEST_DATA_PATH, TRAINING_DATA_PATH])

    train_data = pd.read_csv(TRAINING_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    # Survival rate by sex
    # female_survival_rate = get_feature_rate(train_data, "Sex", "female", "Survived")
    # male_survival_rate = get_feature_rate(train_data, "Sex", "male", "Survived")

    train_predict(train_data, test_data)
