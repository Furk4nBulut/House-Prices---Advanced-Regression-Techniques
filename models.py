import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor  # Uncomment if CatBoost is used

class ModelEvaluator:
    def __init__(self):
        # Initialize the models list
        self.models = [
            ('LR', LinearRegression()),
            # ("Ridge", Ridge()),
            # ("Lasso", Lasso()),
            # ("ElasticNet", ElasticNet()),
            ('KNN', KNeighborsRegressor()),
            ('CART', DecisionTreeRegressor()),
            ('RF', RandomForestRegressor()),
            # ('SVR', SVR()),  # Uncomment if SVR is used
            ('GBM', GradientBoostingRegressor()),
            ("XGBoost", XGBRegressor(objective='reg:squarederror')),
            ("LightGBM", LGBMRegressor())
            # ("CatBoost", CatBoostRegressor(verbose=False))  # Uncomment if CatBoost is used
        ]

    def evaluate_models(self, X, y):
        """
        Evaluates the models using cross-validation and prints RMSE.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
        for name, regressor in self.models:
            rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
            print(f"RMSE: {round(rmse, 4)} ({name}) ")

