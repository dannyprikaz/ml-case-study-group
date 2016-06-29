import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn

class Features(object):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.clf = None
        self.columns = np.array(['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver',
                       'avg_surge', 'signup_date', 'surge_pct',
                       'trips_in_first_30_days', 'weekday_pct',
                       'churn'])

    def feature_pipes(self):
        self.clf = Pipeline([('feature_selection', self.model),
                       ('classification', self.model)])
        self.clf.fit(self.X, self.y)


    def feature_import(self, show=True):

        importances = self.model.feature_importances_

        std = np.std([mod.feature_importances_ for mod in self.model.estimators_],
                     axis=0)

        indices = np.argsort(importances)[::-1]

        # Print feature ranking
        print('Feature ranking:')

        for f in range(self.X.shape[1]):
            print('%d. %s (%f)' % (f + 1, self.columns[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the model
        plt.figure()
        plt.title('Feature importances')
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.barh(range(self.X.shape[1]), importances[indices],
               ecolor='r', xerr=std[indices], align='center')
        plt.yticks(range(self.X.shape[1]), self.columns[indices])
        plt.ylim([-1, self.X.shape[1]])

        if show:
            plt.savefig('feature_importances.png')
            plt.show()
