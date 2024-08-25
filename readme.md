# [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques))

## Overview

Welcome to the repository for the Kaggle "House Prices - Advanced Regression Techniques" competition. This project aims to predict the sale prices of homes based on a comprehensive set of features. The repository is organized to help you understand the data, preprocess it, build and evaluate models, and conduct research for improving predictions.
#### [Kaggle Notebook Solution](https://www.kaggle.com/code/furkanbt/house-prices-advanced-regression-techniques)
## Table of Contents

- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Dataset](#dataset)
- [Helpers](#helperspy)
- [Main Script](#main)
- [Models](#models)
- [Research](#research)
- [Acknowledgments](#acknowledgments)

## Project Structure

This project is organized into the following key components:

- `config/`: Contains configuration files for setting up environment variables and parameters.
- `data/`: Includes raw and processed datasets.
- `datapreprocessing/`: Scripts for cleaning, transforming, and preparing data for modeling.
- `dataset/`: Contains the raw dataset files used in the competition.
- `helpers.py`: Utility functions for various tasks such as data loading, feature engineering, and model evaluation.
- `main.py`: The main script for running the entire pipeline from data preprocessing to model training and evaluation.
- `models/`: Scripts and notebooks related to model building, training, and evaluation.
- `research/`: Jupyter Notebooks for exploratory data analysis, feature engineering experiments, and model research.

## Data Preprocessing

The `datapreprocessing/` directory contains scripts that handle data cleaning, feature engineering, and transformation. Key preprocessing steps include:

1. **Handling Missing Values**: Imputation strategies for different types of missing data.
2. **Feature Engineering**: Creating new features based on domain knowledge and exploratory data analysis.
3. **Data Normalization**: Scaling numerical features and encoding categorical variables.

## Dataset

The `dataset/` directory contains the raw data files provided for the competition. The primary dataset files are:

- `train.csv`: Training data including both features and target values.
- `test.csv`: Test data with features only for which predictions are required.

## Helpers (`helpers.py`)

The `helpers.py` file contains utility functions used across the project. These functions include:

- **Data Loading**: Functions to read and preprocess data files.
- **Feature Engineering**: Functions for creating and transforming features.
- **Model Evaluation**: Functions to evaluate model performance using metrics like RMSE.

## Main Script (`main.py`)

The `main.py` script is the entry point for running the entire pipeline. It includes:

- **Data Loading**: Reading data from CSV files.
- **Data Preprocessing**: Applying transformations and feature engineering.
- **Model Training**: Training various regression models.
- **Model Evaluation**: Assessing model performance and generating predictions.

## Models

The `models/` directory contains scripts and notebooks for building and evaluating different regression models. This includes:

- **Linear Regression**: Baseline model for comparison.
- **Random Forest**: Ensemble model to capture complex interactions.
- **Gradient Boosting**: Advanced model for improved accuracy.
- **XGBoost**: Optimized gradient boosting for better performance.

## Research

The `research/` directory includes Jupyter Notebooks for in-depth analysis and experimentation:

- **Exploratory Data Analysis (EDA)**: Analyzing distributions, correlations, and patterns in the data.
- **Feature Engineering Experiments**: Testing different feature engineering techniques.
- **Model Tuning**: Experimenting with hyperparameter tuning and ensemble methods.

## Acknowledgments

This project uses the Ames Housing dataset, compiled by Dean De Cock, and is part of the Kaggle "House Prices - Advanced Regression Techniques" competition. Special thanks to the Kaggle community and various tutorials that have provided valuable insights and guidance.

## Getting Started

To get started with this project:

1. **Clone the Repository**: `git clone https://github.com/Furk4nBulut/House-Prices-Advanced-Regression-Techniques`
2. **Run the Main Script**: Execute `main.py` to start the data processing and model training pipeline.
3. **Explore Research Notebooks**: Use Jupyter Notebooks in the `research/` directory to review data exploration and model experiments.

Feel free to contribute to the project or open issues for any questions or feedback.
