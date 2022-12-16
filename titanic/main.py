"""
https://www.kaggle.com/competitions/titanic/
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


OUTPUT_DIR_PATH = "./output"
TEST_DATA_PATH = "./data/test.csv"
TRAINING_DATA_PATH = "./data/train.csv"


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
    train_data = pd.read_csv(TRAINING_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    train_predict(train_data, test_data)
