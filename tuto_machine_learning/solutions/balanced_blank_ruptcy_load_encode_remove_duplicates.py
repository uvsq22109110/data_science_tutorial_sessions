from sklearn.preprocessing import LabelEncoder

balanced_bank_ruptcy_label_encoded = pd.read_excel("https://github.com/F3kih/Course_DS/blob/master/balancedRuptcy.xls?raw=true").apply(LabelEncoder().fit_transform)
balanced_bank_ruptcy_label_encoded = balanced_bank_ruptcy_label_encoded.drop_duplicates()
balanced_bank_ruptcy_label_encoded.head()