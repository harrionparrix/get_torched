import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if depth == self.max_depth or num_samples <= 1:
            return np.mean(y)

        # Find the best split
        best_feature = None
        best_split_value = None
        best_loss = float('inf')
        for feature in range(num_features):
            feature_values = X[:, feature]
            for split_value in np.unique(feature_values):
                left_indices = np.where(feature_values <= split_value)[0]
                right_indices = np.where(feature_values > split_value)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_y = y[left_indices]
                right_y = y[right_indices]
                loss = self._calculate_loss(left_y, right_y)
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_split_value = split_value

        if best_feature is None:
            return np.mean(y)

        left_indices = np.where(X[:, best_feature] <= best_split_value)[0]
        right_indices = np.where(X[:, best_feature] > best_split_value)[0]

        left_tree = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)

        return (best_feature, best_split_value, left_tree, right_tree)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if isinstance(tree, np.float64):
            return tree
        else:
            feature, split_value, left_tree, right_tree = tree
            if x[feature] <= split_value:
                return self._predict_single(x, left_tree)
            else:
                return self._predict_single(x, right_tree)

    def _calculate_loss(self, left_y, right_y):
        return np.sum((left_y - np.mean(left_y))**2) + np.sum((right_y - np.mean(right_y))**2)

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        y_pred = np.full_like(y, np.mean(y))  # Initial prediction: mean of y
        for _ in range(self.n_estimators):
            gradient = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, gradient)
            self.trees.append(tree)
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Load the dataset
    boston = fetch_openml(name="house_prices", as_frame=True)
    X = boston.data
    y = boston.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the gradient boosting regressor
    gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb_regressor.fit(X_train, y_train)

    # Predict
    y_pred_train = gb_regressor.predict(X_train)
    y_pred_test = gb_regressor.predict(X_test)

    # Evaluate
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print("Train MSE:", mse_train)
    print("Test MSE:", mse_test)
