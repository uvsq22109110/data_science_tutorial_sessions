le = preprocessing.LabelEncoder()
df_finance["encoded_class"] = le.fit_transform(df_finance.sentiment)

x_train, x_test, y_train, y_test = train_test_split(df_finance.news_vectors,df_finance.encoded_class,test_size=0.2,stratify=df_finance.encoded_class,random_state=42)

clf = GradientBoostingClassifier(n_estimators=85,learning_rate=0.1, max_depth=4, random_state=42)
clf.fit(list(x_train), y_train)
y_pred = clf.predict(list(x_test))
print(clf.score(list(x_train), y_train),clf.score(list(x_test), y_test))
print(classification_report(y_test, y_pred, target_names=le.classes_))