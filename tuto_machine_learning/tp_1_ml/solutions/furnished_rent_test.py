#lenom du test est : Anova one-way
# Ecrire Code et interprétation :

stats.f_oneway(df_housing[df_housing["is furnished"]==1]["Rent(€)"],df_housing[df_housing["is furnished"]==0]["Rent(€)"])

# Inteprétation : Il ya une relation ou pas ? et pouquoi ?  l'hypothèse nulle dit que les deux populations ont 
# la même moyenne et donc les variables sont indépendantes. En comparant, la p-value <<< 005 on peut rejeter l'hypothèse
# nulle et donc les deux variables sont dépendantes.