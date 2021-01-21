tunned_classifiers = {
                        "RF" :  RandomForestClassifier(random_state=2020),
                        "AdaBoost" :  AdaBoostClassifier(random_state=2020),
                        "SVM" :  SVC(probability=True,random_state=2020)
                      }

#setting arameters to eahc classifiers
for clf_name in tunned_classifers_parameters:

    parameters = tunned_classifers_parameters[clf_name]
    tunned_classifiers[clf_name].set_params(**parameters)
    
    
list_tuned_classifiers = (

    ('RF', tunned_classifiers["RF"]),
    ('AdaBoost', tunned_classifiers["AdaBoost"]),
    ('SVM', tunned_classifiers["SVM"]),

)
voting_classifier = VotingClassifier(list_tuned_classifiers,voting="soft")
voting_classifier.fit(x_train,y_train)


y_preds ={}
y_preds["voting"] = voting_classifier.predict(x_test)


for clf_name in tunned_classifers_parameters:

    parameters = tunned_classifers_parameters[clf_name]
    tunned_classifiers[clf_name].set_params(**parameters).fit(x_train,y_train)
    

y_preds ={}
y_preds["voting"] = voting_classifier.predict(x_test)

tunned_classifiers = {
                        "RF" :  RandomForestClassifier(random_state=2020),
                        "AdaBoost" :  AdaBoostClassifier(random_state=2020),
                        "SVM" :  SVC(probability=True,random_state=2020)
                      }

for clf_name in tunned_classifers_parameters:

    parameters = tunned_classifers_parameters[clf_name]
    tunned_classifiers[clf_name].set_params(**parameters).fit(x_train,y_train)
    y_preds[clf_name] = tunned_classifiers[clf_name].predict(x_test)
    

    
print(y_preds.keys())