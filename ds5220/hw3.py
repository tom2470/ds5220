import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
#1a)
df=pd.read_csv('AirQualityUCI.csv')
df=df.drop(columns=['Date','Time', 'X', 'X.1'])
df=df.dropna()
x=df.drop(columns=['HourlyCO'])
y=df.HourlyCO
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#1b)
lr=LinearRegression()
lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)
print("the mean squared error for linear regression is " + str(mean_squared_error(y_test,y_pred)))
#1c)
degree =2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(x_train)
X_poly_test = poly_features.transform(x_test)
poly_model= LinearRegression()
poly_model.fit(X_poly,y_train)
y_poly_fit=poly_model.predict(X_poly_test)
MSE=mean_squared_error(y_test,y_poly_fit)
print("the mean squared error for polynomial degree 2 is " + str(MSE))
#1d)
degree =3
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(x_train)
X_poly_test = poly_features.transform(x_test)
poly_model= LinearRegression()
poly_model.fit(X_poly,y_train)
y_poly_fit=poly_model.predict(X_poly_test)
MSE=mean_squared_error(y_test,y_poly_fit)
print("the mean squared error for polynomial degree 3 is " + str(MSE))
#2a)
#ridge regularization
#-all variables stay in model
#-high correlation among features
#lasso regularization
#-eliminates variables
#-may have many irrelevant or redundant features exist in your data
#elastic net regression
#-eliminates variables
#-can handle situations where Lasso might select only one feature from a group of highly correlated features, as it introduces the Ridge penalty to help keep related features together
#2b)
#standardizon of the variables because then all the coefficents are on the same scale this is because the models are sensitive to the magitude of the coefficents which is dependent on variable scale. 
#2c)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
alpha=0.1
param_grid = {'alpha': alphas}
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
lasso_model = Lasso(alpha=alpha)
grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']
print("The best alpha for lasso is "+ str(best_alpha))
lasso_model = Lasso(alpha=best_alpha)
lasso_model.fit(X_train_scaled, y_train)
y_pred = lasso_model.predict(X_test_scaled)
coeff_df = pd.DataFrame(df.columns)
coeff_df.columns = ["Feature"]
coeff_df["Coefficient Estimate"] = pd.Series(lasso_model.coef_)
print(coeff_df)
MSE=mean_squared_error(y_test,y_pred)
print("The MSE for lasso is "+ str(MSE))
#2d)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
alpha=0.1
param_grid = {'alpha': alphas}
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
Ridge_model = Ridge(alpha=alpha)
grid_search = GridSearchCV(Ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']
print("The best alpha for ridge is "+ str(best_alpha))
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_scaled, y_train)
y_pred = ridge_model.predict(X_test_scaled)
coeff_df = pd.DataFrame(df.columns)
coeff_df.columns = ["Feature"]
coeff_df["Coefficient Estimate"] = pd.Series(ridge_model.coef_)
print(coeff_df)
MSE=mean_squared_error(y_test,y_pred)
print("The MSE for ridge is "+ str(MSE))
#2e)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
l1_ratio = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
alpha=0.1
l1_ratio = 0.3
param_grid = {'alpha': alphas}
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
elastic_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
grid_search = GridSearchCV(elastic_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']
best_l1 = grid_search.best_params_['alpha']
print("The best alpha for elastis net is "+ str(best_alpha))
print("The best l1_ratio is "+ str(best_l1))
elastic_model = ElasticNet(alpha=best_alpha,l1_ratio=best_l1)
elastic_model.fit(X_train_scaled, y_train)
y_pred = elastic_model.predict(X_test_scaled)
coeff_df = pd.DataFrame(df.columns)
coeff_df.columns = ["Feature"]
coeff_df["Coefficient Estimate"] = pd.Series(elastic_model.coef_)
print(coeff_df)
MSE=mean_squared_error(y_test,y_pred)
print("The MSE for elastic net is "+ str(MSE))
#3)
def ridge_gradient_descent(X_b,y,theta,learning_rate,num_iterations,alpha):
    m=len(y)
    for i in range(num_iterations):
        predictions=X_b.dot(theta)
        error=predictions-y
        gradient=(2/m)*(X_b.T.dot(error) +alpha*theta)
        theta=theta-learning_rate*gradient
    return theta

def compute_cost(predictions,y,alpha,theta):
    m =len(y)
    error =predictions-y
    cost=(1/(2*m))*(np.sum(error**2)+alpha*np.sum(theta[1:]**2))
    return cost
theta_initial = np.zeros(X_train_scaled.shape[1])
learning_rate = 0.01
num_iterations = 1000
best_alpha = grid_search.best_params_['alpha']
theta_grad = ridge_gradient_descent(X_train_scaled, y_train, theta_initial, learning_rate, num_iterations, best_alpha)
y_pred_grad = X_test_scaled.dot(theta_grad)
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_scaled, y_train)
y_pred_sklearn = ridge_model.predict(X_test_scaled)
mse_grad = mean_squared_error(y_test,y_pred_grad)
mse_norm = mean_squared_error(y_test,y_pred_sklearn)
print("Ridge Regression MSE " + str(mse_grad))
print("Scikit-learn Ridge Regression MSE "+ str(mse_norm))