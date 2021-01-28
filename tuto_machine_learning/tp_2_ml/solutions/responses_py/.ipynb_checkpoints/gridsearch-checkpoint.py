from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


x_test_embedding = vectorizer.transform(X_test)

# Creating Scoring Function
f1_scorer = make_scorer(f1_score) 

# Creating Models dict
classifiers = {
    
               "MLPClassifier" : MLPClassifier(random_state=2020),
               "RF" : RandomForestClassifier(random_state = 2020), 

              }
# Fixing hyper-parameters 
hyper_parameters = {
                    "RF": {
                             "n_estimators" : [50, 100 , 200, 400],
                             "criterion" : ["entropy", "gini"],
                             "max_depth" : [20, 30, 50]
                    } ,
                    "MLPClassifier": {
                             "hidden_layer_sizes" : [[200, 120,80],[50, 30,20],[100, 70,50]],
                             "learning_rate" : ["constant", "invscaling", "adaptive"] ,
                    },     
                   }



# Tunning hper-parameters
tunned_classifers_parameters = {}
for clf_name in classifiers:

    print("For "+clf_name+" :" )
    clf_to_tune = classifiers[clf_name]
    hyper_parameters_for_clf = hyper_parameters[clf_name]

    clf = GridSearchCV(clf_to_tune, hyper_parameters_for_clf, cv=4, verbose=0, scoring = f1_scorer)
    best_model = clf.fit(x_train_embedding,y_train)
    #Print all the Parameters that gave the best results:
    print('Best Parameters',clf.best_params_)
        
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        
    tunned_classifers_parameters[clf_name] = clf.best_params_
    
print("best parameters",tunned_classifers_parameters)

mlp_Classifier = MLPClassifier(hidden_layer_sizes=(100, 70, 50), learning_rate= "constant", random_state=2020)
rf_classifier =  RandomForestClassifier(criterion= 'entropy', max_depth= 50, n_estimators= 400, random_state = 2020)

mlp_Classifier.fit(x_train_embedding,y_train)
rf_classifier.fit(x_train_embedding,y_train)

y_test_pred_mlp = mlp_Classifier.predict(x_test_embedding)
y_test_pred_rf = rf_classifier.predict(x_test_embedding)

print(classification_report(y_test, y_test_pred_mlp, target_names=le.classes_))
print(classification_report(y_test, y_test_pred_rf, target_names=le.classes_))