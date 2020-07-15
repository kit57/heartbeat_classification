

from sklearn.ensemble import RandomForestRegressor


class RandomForest_model():

    def __init__(self, X_train, y_train, X_val, y_val, epoch, batch_size, hidden_size):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def RF_model(self):

        # Instantiate model with 1000 decision trees
        rf = RandomForestRegressor(max_depth= 8,
                                   max_features= 5,
                                   min_samples_split=5,
                                   n_estimators=500)
        # Train the model on training data
        rf.fit(self.X_train, self.y_train)
