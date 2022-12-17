"""
https://www.kaggle.com/competitions/titanic/
"""
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


OUTPUT_DIR_PATH = "./output"
TEST_DATA_PATH = "./data/test.csv"
TRAINING_DATA_PATH = "./data/train.csv"


def explore_input_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Show various information about the data sets
    :param train_data: DataFrame containing training data set
    :param test_data: DataFrame containing test data set
    """
    print(train_data.head())
    print(test_data.head())

    print("\nMissing values:")
    print(train_data.isnull().sum())
    # sns.heatmap(train_data.isnull(), cbar=False).set_title("Missing values heatmap")
    # matplotlib.pyplot.show()

    print("\nUnique values:")
    print(train_data.nunique())


    # Survived
    num_survived = len(train_data.loc[train_data["Survived"] == 1])
    num_not_survived = len(train_data.loc[train_data["Survived"] == 0])
    survival_rate = num_survived / (num_survived + num_not_survived)
    print("All\t\tsurvival rate:\t", survival_rate)

    # Pclass
    [show_survival_rate(train_data, "Pclass", i) for i in [1, 2, 3]]

    # Sex
    [show_survival_rate(train_data, "Sex", sex) for sex in ["female", "male"]]

    # Age
    # sns.histplot(train_data, x="Age", hue="Survived", binwidth=1)
    # matplotlib.pyplot.show()

    # SibSp
    # sns.histplot(train_data, x="SibSp", hue="Survived", binwidth=1)
    # matplotlib.pyplot.show()

    # Parch
    # sns.histplot(train_data, x="Parch", hue="Survived", binwidth=1)
    # matplotlib.pyplot.show()

    # Ticket
    # print(train_data["Ticket"])

    # Fare
    # sns.histplot(train_data, x="Fare", hue="Survived", binwidth=20)
    # matplotlib.pyplot.show()

    # Embarked
    # sns.histplot(train_data, x="Embarked", hue="Survived")
    # matplotlib.pyplot.show()


def show_survival_rate(
    dataset: pd.DataFrame, feature_header: str, feature_value: str 
):
    """Get survival rate for a given feature
    :param dataset: DataFrame containing input data
    :param feature_header: feature header from data set
    :param feature_value: feature value from data set
    """
    feature = dataset.loc[dataset[feature_header] == feature_value]["Survived"]
    rate = sum(feature) / len(feature)
    print(
            f"{feature_header} {feature_value}\tsurvival rate:\t",
        rate,
    )


def train_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir_path: str = OUTPUT_DIR_PATH,
) -> None:
    """Train and predict survival
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

    explore_input_data(train_data, test_data)

    # train_predict(train_data, test_data)
