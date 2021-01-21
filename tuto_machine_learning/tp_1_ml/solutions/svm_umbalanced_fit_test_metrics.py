from sklearn.svm import LinearSVC
from sklearn.metrics import *

clf = LinearSVC(random_state=2020)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred).round(5)
rec = recall_score(y_test,y_pred).round(5)
pre = precision_score(y_test,y_pred).round(5)

# Commentaire sur les metrics : Recall 1.0, c'est à dire la capacité de distinguer la classe majoritaire parfaitement 
# Précision très haute car on prédit tout comme étant classe majoritaire 
print("Accuracy = {}, Precision = {}, Recall = {}".format(acc,pre,rec))
