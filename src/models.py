import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

class ModelTrainer:
    def __init__(self, model_type='knn'):
        self.model_type = model_type
        self.model = None

    def train(self, X_train, y_train):
        if self.model_type == 'knn':
            print("\n[INFO] Tuning KNN...")
            param_grid = {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            print(f"Best KNN params: {grid.best_params_}")

        elif self.model_type == 'decision_tree':
            print("\n[INFO] Tuning Decision Tree...")
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 5]
            }
            grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            print(f"Best Tree params: {grid.best_params_}")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện!")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print(f"\n--- {self.model_type.upper()} Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        return y_pred