import pandas as pd
import numpy as np
from feature import Features
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from master_clean import load_churn

df = load_churn()

cols = np.array(['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver',
               'avg_surge', 'signup_date', 'surge_pct',
               'trips_in_first_30_days', 'weekday_pct',
               'churn'])

X = df[cols[:-1]].as_matrix()
y = df[cols[-1:]].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

model = RandomForestClassifier(n_estimators=50,
                                oob_score=True)
model.fit(X_train, y_train)

model2 = AdaBoostClassifier(DecisionTreeClassifier(),
                             learning_rate=1,
                             n_estimators=100,
                             random_state=1)
model2.fit(X_train, y_train)

f = Features(model, X_train, y_train)
f2 = Features(model2, X_train, y_train)
f.feature_import()
# f2.feature_import()
# print f.score(X_test, y_test)
