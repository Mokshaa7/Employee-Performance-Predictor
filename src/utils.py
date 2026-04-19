import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, X):
    importance = pd.Series(model.feature_importances_, index=X.columns)

    plt.figure()
    importance.sort_values().plot(kind='barh')
    plt.title("Feature Importance")
    plt.savefig("outputs/feature_importance.png")
    plt.close()