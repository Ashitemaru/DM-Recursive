import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from utils import load_dataset

CATEGORY = "news"
NUM_BINS = 5
MODEL = "random_forest"

def main():
    X, y = load_dataset(CATEGORY)
    
    if CATEGORY == "news":
        # Devide into bins to turn it into classification problem
        y_max = y.max().item()
        y_min = y.min().item()
        y = ((y - y_min) / (y_max - y_min) * NUM_BINS).astype(int)
    
        # SMOTE
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)
        
    # Count the number of each class
    print(y.value_counts())
    
    # K-fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X.iloc[train_index].to_numpy(), X.iloc[test_index].to_numpy()
        y_train, y_test = y.iloc[train_index].to_numpy().ravel(), y.iloc[test_index].to_numpy().ravel()
        
        if MODEL == "linear":
            model = LogisticRegression(multi_class="auto")
        elif MODEL == "random_forest":
            model = RandomForestClassifier(n_estimators=10, criterion="gini")
        else:
            raise ValueError
        
        model.fit(X_train, y_train)
        print(f"Fold {i}: Training over")
        
        # Save the acc
        y_pred = model.predict(X_test)
        acc = sum(y_pred == y_test) / len(y_test)
        accuracies.append(acc)
        
        # Plot the confusion matrix & save to the image folder
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True)
        plt.savefig(f"../image/{MODEL}_{CATEGORY}_{i}.png")
        plt.clf()
    
    print(f"Accuracy: {sum(accuracies) / len(accuracies)}")

if __name__ == "__main__":
    main()