# Nom du test : Test de Student (Test t) : H0 les deux variables continues sont indépendantes, p-val <<< 0.05 H0 rejected 
# Dependency prooved :
print(stats.ttest_ind(df_housing["Area"],df_housing["Rent(€)"]))
print(np.corrcoef(df_housing["Area"],df_housing["Rent(€)"])[0,1])
# 0.96, corrélation très grande, ce qui confirme la linéarité de notre problème
plt.figure(figsize=(12,12))
sns.regplot(x='Area',y='Rent(€)',data=df_housing,truncate =True)
ax.legend(loc="best")
plt.title('Varation of the Rent Amount', bbox={'facecolor':'0.8', 'pad':5},fontsize=8)
plt.plot()