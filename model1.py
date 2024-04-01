from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from config import config
from data import get_data
import wandb

import numpy as np
import json
import matplotlib.pyplot as plt

wandb.init(project = "Mlops_lesson4_logreg")

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    
    coefficients = model.coef_
    coefficients_dict = {}
    for idx, coefficients in enumerate(coefficients):
        coefficients_dict[f"class_{idx}_coefficients"] = coefficients.tolist()

    intercept = model.intercept_
    intercept_list = intercept.tolist()
    regularization = model.C

    model_data = {
        "coefficients": coefficients_dict,
        "intercept": intercept_list,
        "regularization": regularization
    }
    json_file_path = "model_data_LR.json"
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

    image_path = "сonfusion_matrix_LR.png"
    plt.savefig(image_path)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    y_pred_proba = model.predict_proba(x_test)
    roc_auc_scores = []
    for class_index in np.unique(y_test):
        roc_auc_scores.append(roc_auc_score(y_test == class_index, y_pred_proba[:, class_index]))
    mean_roc_auc = np.mean(roc_auc_scores)

    artifact = wandb.Artifact("model_LR_params", type="json")
    artifact.add_file(json_file_path)
    wandb.log_artifact(artifact)
    
    artifact2 = wandb.Artifact("model_LR_confusion_matrix", type="png")
    artifact2.add_file(image_path)
    wandb.log_artifact(artifact2)

    wandb.summary["LR Accuracy"] = accuracy
    wandb.summary["LR F1 score"] = f1
    wandb.summary["LR AUC-ROC"] = mean_roc_auc
    
    wandb.finish()
    print(accuracy_score(y_true=y_test, y_pred=y_pred))


if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])
