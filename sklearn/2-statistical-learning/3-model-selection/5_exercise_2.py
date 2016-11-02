# Exercise 2
# On the diabetes dataset, find the optimal regularization parameter alpha.

# My Attempt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

diabetes = datasets.load_diabetes()

X = diabetes.data[:150]
y = diabetes.target[:150]
alphas = np.logspace(-4,-0.5,30)

scores = list()
best_score, best_alpha = 0, 0
for alpha in alphas:
  clf = Lasso(alpha=alpha)
  this_scores = cross_val_score(clf, X, y, n_jobs=1)
  if np.mean(this_scores) > best_score:
    best_score, best_alpha = np.mean(this_scores), alpha

best_alpha # equals 0.07880...
best_score # equals 0.38753...

# Bonus Question
# How much can you trust the selection of alpha?
lasso_cv = LassoCV(alphas=alphas)
k_fold = KFold(3)
for test, train in k_fold.split(X,y):
  lasso_cv.fit(X[train], y[train])
  print("alpha: %f, score: %f" % (lasso_cv.alpha_, lasso_cv.score(X[test],y[test])))



# Solution
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)
scores = list()
scores_std = list()
n_folds = 3
for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
scores, scores_std = np.array(scores), np.array(scores_std)
# plt.figure().set_size_inches(8, 6)
# plt.semilogx(alphas, scores)
# # plot error lines showing +/- std. errors of the scores
# std_error = scores_std / np.sqrt(n_folds)
# plt.semilogx(alphas, scores + std_error, 'b--')
# plt.semilogx(alphas, scores - std_error, 'b--')
# # alpha=0.2 controls the translucency of the fill color
# plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
# plt.ylabel('CV score +/- std error')
# plt.xlabel('alpha')
# plt.axhline(np.max(scores), linestyle='--', color='.5')
# plt.xlim([alphas[0], alphas[-1]])
# plt.show()



