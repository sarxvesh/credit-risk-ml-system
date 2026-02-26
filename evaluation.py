from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(y_test, pred):
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))