import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import helpers

class DataPreprocessing:
    def __init__(self, dataframe):
        self.df = dataframe

    def preprocess(self):
        # Handle outliers
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if helpers.check_outlier(self.df, col):
                self.df = helpers.replace_with_thresholds(self.df, col)

        # Handle missing values
        self.handle_missing_values()

        # Feature engineering
        self.feature_engineering()

        # Drop unnecessary columns
        self.drop_unnecessary_columns()

        # Label Encoding & One-Hot Encoding
        self.encode_features()

        # Split the data
        return self.split_data()

    def handle_missing_values(self):
        # Replace NaNs with 'No' for specific columns
        no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
                   "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
        for col in no_cols:
            self.df[col].fillna("No", inplace=True)

        # Apply quick missing value imputation
        self.df = helpers.quick_missing_imp(self.df, num_method="median", cat_length=17)

        # Display missing values after imputation
        helpers.missing_values_table(self.df)

    def feature_engineering(self):
        # Creating new features
        self.df["NEW_1st*GrLiv"] = self.df["1stFlrSF"] * self.df["GrLivArea"]
        self.df["NEW_Garage*GrLiv"] = self.df["GarageArea"] * self.df["GrLivArea"]
        self.df["TotalQual"] = self.df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond",
                                        "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional",
                                        "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum()
        self.df["NEW_TotalFlrSF"] = self.df["1stFlrSF"] + self.df["2ndFlrSF"]
        self.df["NEW_TotalBsmtFin"] = self.df["BsmtFinSF1"] + self.df["BsmtFinSF2"]
        self.df["NEW_PorchArea"] = self.df["OpenPorchSF"] + self.df["EnclosedPorch"] + self.df["ScreenPorch"] + self.df["3SsnPorch"] + self.df["WoodDeckSF"]
        self.df["NEW_TotalHouseArea"] = self.df["NEW_TotalFlrSF"] + self.df["TotalBsmtSF"]
        self.df["NEW_TotalSqFeet"] = self.df["GrLivArea"] + self.df["TotalBsmtSF"]
        self.df["NEW_LotRatio"] = self.df["GrLivArea"] / self.df["LotArea"]
        self.df["NEW_RatioArea"] = self.df["NEW_TotalHouseArea"] / self.df["LotArea"]
        self.df["NEW_GarageLotRatio"] = self.df["GarageArea"] / self.df["LotArea"]
        self.df["NEW_MasVnrRatio"] = self.df["MasVnrArea"] / self.df["NEW_TotalHouseArea"]
        self.df["NEW_DifArea"] = self.df["LotArea"] - self.df["1stFlrSF"] - self.df["GarageArea"] - self.df["NEW_PorchArea"] - self.df["WoodDeckSF"]
        self.df["NEW_OverallGrade"] = self.df["OverallQual"] * self.df["OverallCond"]
        self.df["NEW_Restoration"] = self.df["YearRemodAdd"] - self.df["YearBuilt"]
        self.df["NEW_HouseAge"] = self.df["YrSold"] - self.df["YearBuilt"]
        self.df["NEW_RestorationAge"] = self.df["YrSold"] - self.df["YearRemodAdd"]
        self.df["NEW_GarageAge"] = self.df["GarageYrBlt"] - self.df["YearBuilt"]
        self.df["NEW_GarageRestorationAge"] = np.abs(self.df["GarageYrBlt"] - self.df["YearRemodAdd"])
        self.df["NEW_GarageSold"] = self.df["YrSold"] - self.df["GarageYrBlt"]

    def drop_unnecessary_columns(self):
        drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Heating", "PoolQC", "MiscFeature", "Neighborhood"]
        self.df.drop(drop_list, axis=1, inplace=True)

    def encode_features(self):
        cat_cols, cat_but_car, num_cols = helpers.grab_col_names(self.df)

        binary_cols = [col for col in self.df.columns if self.df[col].dtypes == "O" and len(self.df[col].unique()) == 2]

        # Apply label encoding to binary columns
        for col in binary_cols:
            self.df = helpers.label_encoder(self.df, col)

        # Apply one-hot encoding to categorical columns
        self.df = helpers.one_hot_encoder(self.df, cat_cols, drop_first=True)

    def split_data(self):
        # Split the dataset into train and test
        train_df = self.df[self.df['SalePrice'].notnull()]
        test_df = self.df[self.df['SalePrice'].isnull()]

        y = train_df['SalePrice']
        X = train_df.drop(["Id", "SalePrice"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

        return X_train, X_test, y_train, y_test
