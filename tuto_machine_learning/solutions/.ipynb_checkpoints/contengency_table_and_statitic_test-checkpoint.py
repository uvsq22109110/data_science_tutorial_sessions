contingency_table = pd.crosstab(index=df_housing['is furnished'],columns=df_housing['Rent from Agency'])
# Khi2 : H0 indépendence entre les variables catégorielles

stats.chi2_contingency(contingency_table)

# Comme p <<< 0.05 on rejete H0, les deux variables sont liées... (Les agences suivent plutôt une politique pour 
# maximiser leurs gains)