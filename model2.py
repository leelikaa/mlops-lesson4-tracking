from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from config import config
from data import get_data
from clearml import Task

import numpy as np
import json
import matplotlib.pyplot as plt


task = Task.init(
    project_name="Mlops_lesson4",
    task_name="Mlops_lesson4_Des_tree")


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер

    depth = int(model.get_depth())
    leaves = int(model.get_n_leaves())
    criterion = model.criterion

    model_data = {
        "Tree depth": depth,
        "Leaves num": leaves,
        "criterion": criterion
    }
    json_file_path = "model_data_DT.json"
    with open(json_file_path, "w") as json_file:
        json.dump(model_data, json_file)

    matrix = confusion_matrix(y_test, y_pred)

    plt.imshow(matrix, interpolation='nearest')
    plt.title('Сonfusion matrix')
    plt.colorbar()

    classes = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9',
               'class 10']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.ylabel('y_test')
    plt.xlabel('y_pred')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(matrix[i, j]), horizontalalignment='center',
                     color='black' if matrix[i, j] > matrix.max() / 2 else 'white')

    image_path = "сonfusion_matrix_DT.png"
    plt.savefig(image_path)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    y_pred_proba = model.predict_proba(x_test)
    roc_auc_scores = []
    for class_index in np.unique(y_test):
        roc_auc_scores.append(roc_auc_score(y_test == class_index, y_pred_proba[:, class_index]))
    mean_roc_auc = np.mean(roc_auc_scores)

    logger = task.get_logger()

    task.upload_artifact(name='model_DT_params', artifact_object=json_file_path)
    task.upload_artifact(name='model_DT_confusion_matrix', artifact_object=image_path)

    logger.report_scalar(title="Accuracy", series="DT_test", value=accuracy, iteration=0)
    logger.report_scalar(title="F1 score", series="DT_test", value=f1, iteration=0)
    logger.report_scalar(title="AUC-ROC", series="DT_test", value=mean_roc_auc, iteration=0)
    print(accuracy_score(y_true=y_test, y_pred=y_pred))


if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"]
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])
