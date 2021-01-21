df_repartition = df_housing.groupby(["Rent from Agency",'is furnished']).size().reset_index(name='counts')
#df_repartition.head()

# Create 1x3 sub plots
gs = gridspec.GridSpec(1, 3)

plt.figure(figsize=(12,12))

ax = plt.subplot(gs[0, 0]) # row 0, col 0
ax.pie([df_repartition[df_repartition["is furnished"]==0]["counts"].sum(),df_repartition[df_repartition["is furnished"]==1]["counts"].sum()]
,labels=["unfurnished","furnished"],colors=["r","b"],autopct='%1.1f%%')
plt.title("All appartments")

ax = plt.subplot(gs[0, 1]) # row 0, col 1
ax.pie([df_repartition[((df_repartition["is furnished"]==0) & (df_repartition["Rent from Agency"]==1))]["counts"].sum(),df_repartition[((df_repartition["is furnished"]==1 ) & (df_repartition["Rent from Agency"]==1))]["counts"].sum()]
,labels=["unfurnished","furnished"],colors=["r","b"],autopct='%1.1f%%')
plt.title("Agencies appartments")

ax = plt.subplot(gs[0, 2]) # row 1, span all columns
ax.pie([df_repartition[((df_repartition["is furnished"]==0) & (df_repartition["Rent from Agency"]==0))]["counts"].sum(),df_repartition[((df_repartition["is furnished"]==1 ) & (df_repartition["Rent from Agency"]==0))]["counts"].sum()]
,labels=["unfurnished","furnished"],colors=["r","b"],autopct='%1.1f%%')

plt.title("Individuals appartments")


plt.plot()
