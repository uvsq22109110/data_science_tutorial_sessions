from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet


x_train = df_housing[df_housing.columns[:-1]].values
y_train = df_housing[df_housing.columns[-1]].values

reg_random_forest = RandomForestRegressor(max_depth=2, random_state=2020,n_estimators=50)
reg_linear_regression = LinearRegression()
reg_elastic_net = ElasticNet(random_state=2020)

reg_random_forest.fit(x_train, y_train)
reg_linear_regression.fit(x_train, y_train)
reg_elastic_net.fit(x_train, y_train)