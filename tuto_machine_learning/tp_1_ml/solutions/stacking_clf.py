class Stacking:
    
    def __init__(self, voting_clf):
        
        from sklearn.neighbors import KNeighborsClassifier
        
        self.knn_clf = KNeighborsClassifier()
        self.voting_clf = voting_clf
    
    def fit(self,x_train_data,y_train_data):
        
        self.voting_clf.fit(x_train_data,y_train_data)
        x_data_knn_train = np.column_stack((x_train_data,self.voting_clf.predict_proba(x_train_data))) 
        self.knn_clf.fit(x_data_knn_train,y_train_data)
        
    def predict(self,x_test_data):
        
        x_data_knn_test = np.column_stack((x_test_data,self.voting_clf.predict_proba(x_test_data))) 
        return self.knn_clf.predict(x_data_knn_test)


stacking_classifer = Stacking(voting_classifier)
stacking_classifer.fit(x_train,y_train)
y_preds["stacking"] = stacking_classifer.predict(x_test)
roc_curve_ploter(y_preds["stacking"], y_test, "stacking")