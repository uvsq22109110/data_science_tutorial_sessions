#Réponse
def affect_to_quarter(month_year):
    
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    [month,year]  = month_year.split()
    index_month = months.index(month[:3])
    trim = year+"-"+"Q"+str(1+index_month//3)
    return trim

us_china["Trim"] = us_china["Month"].apply(lambda x : affect_to_Trim(x))
us_china_trim = us_china.groupby(["Trim"], as_index=False).sum()