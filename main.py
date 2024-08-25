# pipeline
import config
from dataset import DataLoader
from data_preprocessing import DataPreprocessing
from models import ModelEvaluator

def main():
    data = DataLoader(config.TRAINING_DATA_PATH, config.TESTING_DATA_PATH).get_data()

    preprocessor = DataPreprocessing(data)
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    evaluator = ModelEvaluator()
    evaluator.evaluate_models(X_train, y_train)

if __name__ == "__main__":
    main()