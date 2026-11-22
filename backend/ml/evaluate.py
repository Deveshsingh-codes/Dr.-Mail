from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

def evaluate(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "report": classification_report(y_true, y_pred, target_names=["ham", "spam"], zero_division=0)}
