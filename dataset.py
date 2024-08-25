import pandas as pd
import config

class DataLoader:
    def __init__(self, training_data_path, testing_data_path):
        self.training_data_path = training_data_path
        self.testing_data_path = testing_data_path
        print("Initializing DataLoader...")

    def get_data(self):
        print("Loading data...")
        # Load and combine data
        train = pd.read_csv(self.training_data_path)
        test = pd.read_csv(self.testing_data_path)
        df = pd.concat([train, test], ignore_index=True)
        df = df.reset_index(drop=True)
        print("Data loaded successfully.")
        return df


