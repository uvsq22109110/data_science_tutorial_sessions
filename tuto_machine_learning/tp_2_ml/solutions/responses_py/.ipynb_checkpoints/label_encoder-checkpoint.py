le = preprocessing.LabelEncoder()
df["spam_ham_encoded"] = le.fit_transform(df["spam_ham"].values)
print(le.classes_)
X_train, X_test, y_train, y_test = train_test_split(df["text_lower"] , df["spam_ham_encoded"] , test_size=0.2, shuffle=True, stratify=df["spam_ham_encoded"]  ,random_state=42)