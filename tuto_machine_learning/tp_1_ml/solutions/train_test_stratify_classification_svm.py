from sklearn.model_selection import train_test_split

x_data = data_banking_ruptcy_label_encoded[data_banking_ruptcy.columns[:-1]]
y_data = data_banking_ruptcy_label_encoded[data_banking_ruptcy.columns[-1]].values
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, stratify=y_data, random_state=2020)