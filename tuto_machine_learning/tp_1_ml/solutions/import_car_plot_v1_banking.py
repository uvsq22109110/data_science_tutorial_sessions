data_banking_ruptcy = pd.read_excel("https://github.com/F3kih/Course_DS/blob/master/umbalancedRuptcy.xls?raw=true")
print("Before drop duplicates, {} rows in dataframe".format(data_banking_ruptcy.shape[0]))
# droping duplicate rows
data_banking_ruptcy = data_banking_ruptcy.drop_duplicates()
print("After drop duplicates, {} rows in dataframe".format(data_banking_ruptcy.shape[0]))
data_banking_ruptcy['Class'].value_counts().plot(kind='bar',
                                    figsize=(5,5),
                                    title="Number for each Owner Name")


# Remarque : jeu de données non équilibré car il'ya une classe minoritaire < 0.05 % du dataset