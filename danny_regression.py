from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from master_clean import load_churn
import pandas as pd

df = load_churn()

y = df.pop('churn').values
X = df.values

scale = StandardScaler()

scale.fit_transform(X)

param_grid = {'C': [10, 1, 0.1, 0.01, 0.001, 0.0001]}

gs = GridSearchCV(LogisticRegression(), param_grid)

gs.fit(X, y)

print gs.best_params_

# C = 0.01
