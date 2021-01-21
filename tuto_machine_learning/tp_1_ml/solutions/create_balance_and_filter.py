us_china["Balance"] = us_china[["Exports","Imports"]].apply(lambda x : x[0]-x[1], axis=1)
us_china = us_china[us_china["Month"]!="July 2019"]
us_china.tail(2)