from sklearn.preprocessing import LabelEncoder

data_banking_ruptcy_label_encoded = data_banking_ruptcy.apply(LabelEncoder().fit_transform)
data_banking_ruptcy_label_encoded.head()