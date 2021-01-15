def roc_curve_ploter(y_pred, y_test_label, model_name):

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    
    fpr, tpr, _ = roc_curve(y_test_label,  y_pred)
    auc = roc_auc_score(y_test_label, y_pred)
    plt.plot(fpr,tpr,label=model_name+" AUC = "+str(round(auc,4)))
    plt.legend(loc=4)
    plt.show()

for clf_name in y_preds:
    roc_curve_ploter(y_preds[clf_name], y_test, clf_name)