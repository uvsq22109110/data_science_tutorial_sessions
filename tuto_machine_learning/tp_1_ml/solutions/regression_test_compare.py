from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

df_housing_test = pd.read_excel('https://github.com/F3kih/Course_DS/blob/master/test_Regression_Housing_Paris.xlsx?raw=true')

# predictions
x_test = df_housing_test[df_housing_test.columns[:-1]].values
y_test =df_housing_test[df_housing_test.columns[-1]].values
y_pred_rf = reg_random_forest.predict(x_test)
y_pred_lin = reg_linear_regression.predict(x_test)
y_pred_elastic = reg_elastic_net.predict(x_test)

#comparison


print(" Using R^2 score ")
if((r2_score(y_test,y_pred_lin) > r2_score(y_test,y_pred_rf) ) and (r2_score(y_test,y_pred_lin) > r2_score(y_test,y_pred_elastic))) :
    print("Linear regression is the better Model")
elif((r2_score(y_test,y_pred_elastic) > r2_score(y_test,y_pred_rf) ) and (r2_score(y_test,y_pred_elastic) > r2_score(y_test,y_pred_lin))) :
    print("Elastic-net is the better Model")
else:
    print("Random Forest is the better Model")
    
print(" Using Mean squared error score ")    
if((mean_squared_error(y_test,y_pred_lin) < mean_squared_error(y_test,y_pred_rf) ) and (mean_squared_error(y_test,y_pred_lin) < mean_squared_error(y_test,y_pred_elastic))) :
    print("Linear regression is the better Model")
elif((mean_squared_error(y_test,y_pred_elastic) < mean_squared_error(y_test,y_pred_rf) ) and (mean_squared_error(y_test,y_pred_elastic) < mean_squared_error(y_test,y_pred_lin))) :
    print("Elastic-net is the better Model")
else:
    print("Random Forest is the better Model")
    
 