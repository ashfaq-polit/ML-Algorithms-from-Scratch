import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


class MatrixLinearRegression:

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)   # add ones vector
        XT_X_inv = np.linalg.inv(X.T @ X)   # (X.T * X) ** (-1) inverse matrix
        weights = np.linalg.multi_dot([XT_X_inv, X.T, y])   # XT_X_inv * X.T * y
        self.bias, self.weights = weights[0], weights[1:]

    def predict(self, X_test):
        return X_test @ self.weights + self.bias


class GDLinearRegression:
    def __init__(self, learning_rate=0.01, tolerance=1e-8):
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias, self.weights = 0, np.zeros(n_features)
        previous_db, previous_dw = 0, np.zeros(n_features)

        while True:
            y_pred = X @ self.weights + self.bias
            db = 1 / n_samples * np.sum(y_pred - y)
            dw = 1 / n_samples * X.T @ (y_pred - y)
            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw

            abs_db_reduction = np.abs(db - previous_db)
            abs_dw_reduction = np.abs(dw - previous_dw)

            if abs_db_reduction < self.tolerance:
                if abs_dw_reduction.all() < self.tolerance:
                    break

            previous_db = db
            previous_dw = dw

    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    
    
### https://www.kaggle.com/datasets/hussainnasirkhan/multiple-linear-regression-dataset   
df_path = "/Users/adury/Desktop/multiple_linear_regression_dataset.csv"
income = pd.read_csv(df_path)
X1, y1 = income.iloc[:, :-1].values, income.iloc[:, -1].values
X1_scaled = scale(X1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=0)
X1_train_s, X1_test_s, y1_train, y1_test = train_test_split(X1_scaled, y1, random_state=0)
print(income)

correlation_matrix = income.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm')


matrix_linear_regression = MatrixLinearRegression()
matrix_linear_regression.fit(X1_train_s, y1_train)
matrix_lr_pred_res = matrix_linear_regression.predict(X1_test_s)
matrix_lr_r2 = r2_score(y1_test, matrix_lr_pred_res)
matrix_lr_mape = mean_absolute_percentage_error(y1_test, matrix_lr_pred_res)

print(f'Matrix Linear regression  R2 score: {matrix_lr_r2}')
print(f'Matrix Linear regression MAPE: {matrix_lr_mape}', '\n')

print(f'weights: {matrix_linear_regression.bias, *matrix_linear_regression.weights}')
print(f'prediction: {matrix_lr_pred_res}')


linear_regression = GDLinearRegression()
linear_regression.fit(X1_train_s, y1_train)
pred_res = linear_regression.predict(X1_test_s)
r2 = r2_score(y1_test, pred_res)
mape = mean_absolute_percentage_error(y1_test, pred_res)

print(f'Linear regression R2 score: {r2}')
print(f'Linear regression MAPE: {mape}', '\n')

print(f'weights: {linear_regression.bias, *linear_regression.weights}')
print(f'prediction: {pred_res}')



sk_linear_regression = LinearRegression()
sk_linear_regression.fit(X1_train, y1_train)

sk_lr_pred_res = sk_linear_regression.predict(X1_test)
sk_lr_r2 = r2_score(y1_test, sk_lr_pred_res)
sk_lr_mape = mean_absolute_percentage_error(y1_test, sk_lr_pred_res)

print(f'Scikit-learn Linear regression R2 score: {sk_lr_r2}')
print(f'Scikit-learn Linear regression MAPE: {sk_lr_mape}', '\n')

print(f'weights: {sk_linear_regression.intercept_, *sk_linear_regression.coef_}')
print(f'prediction: {sk_lr_pred_res}', '\n')

sk_linear_regression.fit(X1_train_s, y1_train)
print(f'scaled weights: {sk_linear_regression.intercept_, *sk_linear_regression.coef_}')


feature1, feature2 = X1[:, 0], X1[:, 1]
X1_linspace = np.linspace(feature1.min(), feature1.max())
X2_linspace = np.linspace(feature2.min(), feature2.max())
X1_surface, X2_surface = np.meshgrid(X1_linspace, X2_linspace)
X_surfaces = np.array([X1_surface.ravel(), X2_surface.ravel()]).T

sk_linear_regression = LinearRegression()
sk_linear_regression.fit(X1_train, y1_train)
y_surface = sk_linear_regression.predict(X_surfaces).reshape(X1_surface.shape)

fig = plt.figure(figsize=(9, 7))
ax = plt.axes(projection='3d')
ax.scatter(feature1, feature2, y1, color='red', marker='o')
ax.plot_surface(X1_surface, X2_surface, y_surface, color='black', alpha=0.6)
plt.title('Fitted linear regression surface')
ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Income')
ax.view_init(20, 10)
plt.show()


X3, y3 = make_regression(n_samples=14, n_features=1, noise=2, random_state=0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=0)
print('X3', X3, sep='\n')
print('y3', y3, sep='\n')

sk_linear_regression = LinearRegression()
sk_linear_regression.fit(X3_train, y3_train)

sk_lr_pred_res = sk_linear_regression.predict(X3_test)
sk_lr_pred_train_res = sk_linear_regression.predict(X3_train)

sk_lr_r2 = r2_score(y3_test, sk_lr_pred_res)
sk_lr_train_r2 = r2_score(y3_train, sk_lr_pred_train_res)

sk_lr_mse = mean_squared_error(y3_test, sk_lr_pred_res)
sk_lr_train_mse = mean_squared_error(y3_train, sk_lr_pred_train_res)

sk_lr_mape = mean_absolute_percentage_error(y3_test, sk_lr_pred_res)
sk_lr_train_mape = mean_absolute_percentage_error(y3_train, sk_lr_pred_train_res)

print(f'Linear regression R2 score: {sk_lr_r2}')
print(f'Linear regression train R2 score: {sk_lr_train_r2}', '\n')

print(f'Linear regression MSE: {sk_lr_mse}')
print(f'Linear regression train MSE: {sk_lr_train_mse}', '\n')

print(f'Linear regression MAPE: {sk_lr_mape}')
print(f'Linear regression train MAPE: {sk_lr_train_mape}', '\n')

print(f'prediction: {sk_lr_pred_res}')



sk_ridge_regression = Ridge()
sk_ridge_regression.fit(X3_train, y3_train)

sk_ridge_pred_res = sk_ridge_regression.predict(X3_test)
sk_ridge_pred_train_res = sk_ridge_regression.predict(X3_train)

sk_ridge_r2 = r2_score(y3_test, sk_ridge_pred_res)
sk_ridge_train_r2 = r2_score(y3_train, sk_ridge_pred_train_res)

sk_ridge_mse = mean_squared_error(y3_test, sk_ridge_pred_res)
sk_ridge_train_mse = mean_squared_error(y3_train, sk_ridge_pred_train_res)

sk_ridge_mape = mean_absolute_percentage_error(y3_test, sk_ridge_pred_res)
sk_ridge_train_mape = mean_absolute_percentage_error(y3_train, sk_ridge_pred_train_res)

print(f'Ridge R2 score: {sk_ridge_r2}')
print(f'Ridge train R2 score: {sk_ridge_train_r2}', '\n')

print(f'Ridge MSE: {sk_ridge_mse}')
print(f'Ridge train MSE: {sk_ridge_train_mse}', '\n')

print(f'Ridge MAPE: {sk_ridge_mape}')
print(f'Ridge train MAPE: {sk_ridge_train_mape}', '\n')

print(f'prediction: {sk_ridge_pred_res}')


sk_lasso_regression = Lasso()
sk_lasso_regression.fit(X3_train, y3_train)

sk_lasso_pred_res = sk_lasso_regression.predict(X3_test)
sk_lasso_pred_train_res = sk_lasso_regression.predict(X3_train)

sk_lasso_r2 = r2_score(y3_test, sk_lasso_pred_res)
sk_lasso_train_r2 = r2_score(y3_train, sk_lasso_pred_train_res)

sk_lasso_mse = mean_squared_error(y3_test, sk_lasso_pred_res)
sk_lasso_train_mse = mean_squared_error(y3_train, sk_lasso_pred_train_res)

sk_lasso_mape = mean_absolute_percentage_error(y3_test, sk_lasso_pred_res)
sk_lasso_train_mape = mean_absolute_percentage_error(y3_train, sk_lasso_pred_train_res)

print(f'Lasso R2 score: {sk_lasso_r2}')
print(f'Lasso train R2 score: {sk_lasso_train_r2}', '\n')

print(f'Lasso MSE: {sk_lasso_mse}')
print(f'Lasso train MSE: {sk_lasso_train_mse}', '\n')

print(f'Lasso MAPE: {sk_lasso_mape}')
print(f'Lasso train MAPE: {sk_lasso_train_mape}', '\n')

print(f'prediction: {sk_lasso_pred_res}')


sk_elastic_net_regression = ElasticNet()
sk_elastic_net_regression.fit(X3_train, y3_train)

sk_elastic_net_pred_res = sk_elastic_net_regression.predict(X3_test)
sk_elastic_net_pred_train_res = sk_elastic_net_regression.predict(X3_train)

sk_elastic_net_r2 = r2_score(y3_test, sk_elastic_net_pred_res)
sk_elastic_net_train_r2 = r2_score(y3_train, sk_elastic_net_pred_train_res)

sk_elastic_net_mse = mean_squared_error(y3_test, sk_elastic_net_pred_res)
sk_elastic_net_train_mse = mean_squared_error(y3_train, sk_elastic_net_pred_train_res)

sk_elastic_net_mape = mean_absolute_percentage_error(y3_test, sk_elastic_net_pred_res)
sk_elastic_net_train_mape = mean_absolute_percentage_error(y3_train, sk_elastic_net_pred_train_res)

print(f'ElasticNet R2 score: {sk_elastic_net_r2}')
print(f'ElasticNet train R2 score: {sk_elastic_net_train_r2}', '\n')

print(f'ElasticNet MSE: {sk_elastic_net_mse}')
print(f'ElasticNet train MSE: {sk_elastic_net_train_mse}', '\n')

print(f'ElasticNet MAPE: {sk_elastic_net_mape}')
print(f'ElasticNet train MAPE: {sk_elastic_net_train_mape}', '\n')

print(f'prediction: {sk_elastic_net_pred_res}')


train_r2_scores = [sk_lr_train_r2, sk_ridge_train_r2, sk_lasso_train_r2, sk_elastic_net_train_r2]
test_r2_scores = [sk_lr_r2, sk_ridge_r2, sk_lasso_r2, sk_elastic_net_r2]

train_mses = [sk_lr_train_mse, sk_ridge_train_mse, sk_lasso_train_mse, sk_elastic_net_train_mse]
test_mses = [sk_lr_mse, sk_ridge_mse, sk_lasso_mse, sk_elastic_net_mse]

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.title.set_text('R2 reduction')
ax1.set_xlabel('R2 train')
ax1.set_ylabel('test')
ax1.scatter(train_r2_scores, test_r2_scores, color='red')
ax1.plot(train_r2_scores, test_r2_scores)

ax2 = fig.add_subplot(222)
ax2.title.set_text('MSE reduction')
ax2.set_xlabel('MSE train')
ax2.scatter(train_mses, test_mses, color='red')
ax2.plot(train_mses, test_mses)


sk_linear_regression_pred_all_data_res = sk_linear_regression.predict(X3)
sk_ridge_regression_pred_all_data_res = sk_ridge_regression.predict(X3)
sk_lasso_regression_pred_all_data_res = sk_lasso_regression.predict(X3)
sk_elastic_net_regression_all_data_pred_res = sk_elastic_net_regression.predict(X3)

plt.scatter(X3, y3, color='black')
plt.plot(X3, sk_linear_regression_pred_all_data_res, label='Linear regression')
plt.plot(X3, sk_ridge_regression_pred_all_data_res, label='Ridge')
plt.plot(X3, sk_lasso_regression_pred_all_data_res, label='Lasso')
plt.plot(X3, sk_elastic_net_regression_all_data_pred_res, label='ElasticNet')
plt.title('Regression regularizations comparison')
plt.xlabel('X3')
plt.ylabel('y3')
plt.legend()
plt.show()



import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_residuals(X, y, title):
    """A helper function to fit a model and plot residuals."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    residuals = model.resid
    predicted = model.predict()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)

    # Residual vs. Predicted Plot
    ax1.scatter(predicted, residuals, alpha=0.6)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Predicted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residual vs. Predicted Plot")

    # Q-Q Plot for Normality
    sm.qqplot(residuals, line='45', fit=True, ax=ax2)
    ax2.set_title("Q-Q Plot of Residuals")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Set a seed for reproducibility
np.random.seed(42)


# Generate data
X1 = np.linspace(0, 100, 200)
# Error is from a normal distribution with constant variance
error1 = np.random.normal(0, 10, 200)
y1 = 5 + 2 * X1 + error1

plot_residuals(X1, y1, "Case 1: Homoscedastic and Normal")


# Generate data
X2 = np.linspace(1, 100, 200)
# Error is from a normal distribution, but its variance increases with X
error2 = np.random.normal(0, 0.5, 200) * X2
y2 = 5 + 2 * X2 + error2

plot_residuals(X2, y2, "Case 2: Heteroscedastic but Normal")


# Generate data
X3 = np.linspace(0, 100, 200)
# Error is from a skewed (exponential) distribution but has constant variance
error3 = (np.random.exponential(1, 200) - 1) * 15 # Subtract 1 to center, multiply to scale
y3 = 5 + 2 * X3 + error3

plot_residuals(X3, y3, "Case 3: Homoscedastic but Non-Normal")


# Generate data
X4 = np.linspace(1, 100, 200)
# Error is from a skewed distribution AND its variance increases with X
error4 = (np.random.exponential(1, 200) - 1) * X4 * 0.5
y4 = 5 + 2 * X4 + error4

plot_residuals(X4, y4, "Case 4: Heteroscedastic and Non-Normal")



import numpy as np
import statsmodels.api as sm

# Toy dataset
# Let's say we want to predict exam score (y) from study hours (X)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Add a constant (for intercept)
X = sm.add_constant(X)  

# Fit linear regression
model = sm.OLS(y, X).fit()

# Print summary (gives coefficients, standard errors, p-values, confidence intervals)
print(model.summary())



import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Generate heteroscedastic data
np.random.seed(0)
n = 100
X = np.linspace(1, 10, n)
y = 2*X + np.random.normal(0, X, n)  # variance ~ X

X = sm.add_constant(X)

# OLS
ols = sm.OLS(y, X).fit()
print("OLS summary (with wrong SEs):")
print(ols.summary())

# Test for heteroscedasticity
bp_test = het_breuschpagan(ols.resid, X)
print("\nBreusch–Pagan test: LM Stat=%.2f, p=%.4f" % (bp_test[0], bp_test[1]))

# OLS with robust SEs
ols_robust = sm.OLS(y, X).fit(cov_type='HC3')
print("\nOLS with robust SEs:")
print(ols_robust.summary())

# Feasible GLS: estimate weights from residuals
resid_sq = ols.resid**2
aux_reg = sm.OLS(np.log(resid_sq), X).fit()  # variance model
fitted_var = np.exp(aux_reg.fittedvalues)
weights = 1 / fitted_var

gls = sm.WLS(y, X, weights=weights).fit()
print("\nFeasible GLS summary:")
print(gls.summary())



import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# -----------------------------
# 1. Create toy dataset
# -----------------------------
np.random.seed(42)
n = 100
X1 = np.random.rand(n) * 10
X2 = X1 + np.random.normal(0, 0.1, n)  # highly correlated with X1
X3 = np.random.rand(n) * 10
y = 3*X1 + 2*X3 + np.random.normal(0, 1, n)  # true model depends on X1, X3

# Put into DataFrame
df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

# -----------------------------
# 2. Check correlation
# -----------------------------
print(df[['X1','X2','X3']].corr())

# -----------------------------
# 3. OLS Regression
# -----------------------------
X = df[['X1','X2','X3']]
X = sm.add_constant(X)
ols_model = sm.OLS(df['y'], X).fit()
print("\nOLS Regression Summary (with multicollinearity):")
print(ols_model.summary())

# -----------------------------
# 4. Ridge Regression
# -----------------------------
ridge = Ridge(alpha=10)   # alpha = penalty strength
ridge.fit(df[['X1','X2','X3']], y)
print("\nRidge coefficients:", ridge.coef_)

# -----------------------------
# 5. Lasso Regression
# -----------------------------
lasso = Lasso(alpha=0.1)
lasso.fit(df[['X1','X2','X3']], y)
print("\nLasso coefficients:", lasso.coef_)






