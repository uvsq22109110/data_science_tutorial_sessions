from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


# Creating Scoring Function
f1_scorer = make_scorer(f1_score) 


# Split the dataset in two equal parts
x_data = balanced_bank_ruptcy_label_encoded[balanced_bank_ruptcy_label_encoded.columns[:-1]]
y_data = balanced_bank_ruptcy_label_encoded[balanced_bank_ruptcy_label_encoded.columns[-1]].values
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data, random_state=2020)

# Creating Models dict
        
classifiers = {
    
               "RF" : RandomForestClassifier(random_state = 2020), 
               "AdaBoost" : AdaBoostClassifier(random_state = 2020) ,
               "SVM" : SVC(random_state = 2020)
    
              }
# Fixing hyper-parameters 
hyper_parameters = {
                    "RF": {
                             "n_estimators" : [2, 5, 10 , 20],
                             "criterion" : ["entropy", "gini"],
                             "max_depth" : [1, 2, 3]
                    } ,
                    "AdaBoost": {
                             "n_estimators" : [10, 15, 20],
                             "learning_rate" : [0.0001, 0.001, 0.01, 0.1] ,
                    }, 
                    "SVM": {
                             "kernel" : ["rbf","linear"],
                             "C" : [1, 10, 100, 200] ,
                             "gamma" : [1e-2, 1e-3, 1e-4, 1e-5]                                              
                    } 
    
                   }



# Tunning hper-parameters
tunned_classifers_parameters = {}
for clf_name in classifiers:

    print("For "+clf_name+" :" )
    clf_to_tune = classifiers[clf_name]
    hyper_parameters_for_clf = hyper_parameters[clf_name]

    clf = GridSearchCV(clf_to_tune, hyper_parameters_for_clf, cv=5, verbose=0, scoring = f1_scorer)
    best_model = clf.fit(x_train,y_train)
    #Print all the Parameters that gave the best results:
    print('Best Parameters',clf.best_params_)
        
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        
    tunned_classifers_parameters[clf_name] = clf.best_params_