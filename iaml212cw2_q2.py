
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml212cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import pandas as pd
import scipy
from scipy.stats import itemfreq
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture

from iaml_cw2_helpers import *
Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()

Xtrn = Xtrn_org / 255.0
Xtst = Xtst_org / 255.0
Ytrn = Ytrn_org - 1
Ytst = Ytst_org - 1
Xmean = np.mean(Xtrn, axis = 0)
Xtrn_m = Xtrn - Xmean; Xtst_m = Xtst - Xmean # Mean normalised versions
#<----

# Q2.1

def iaml212cw2_q2_1():
    #Q2.1 a)

    #Training Set
    print('Number of instances: {}, number of attributes: {}'.format(Xtrn.shape[0], Xtrn.shape[1]))
    Xtrndf = pd.DataFrame(Xtrn)
    print(np.max(Xtrn), np.min(Xtrn), np.mean(Xtrn), np.std(Xtrn))

    #Testing Set
    print('Number of instances: {}, number of attributes: {}'.format(Xtst.shape[0], Xtst.shape[1]))
    Xtstdf = pd.DataFrame(Xtst)
    print(np.max(Xtst), np.min(Xtst), np.mean(Xtst), np.std(Xtst))

    #Q2.1 b)
    plt.imshow(Xtrn[0].reshape((28,28)).T, cmap="gray_r")
    plt.title(f"Class {Ytrn[0]}")
    plt.savefig("results/2_1_1.png")
    plt.show()
    plt.imshow(Xtrn[1].reshape((28,28)).T, cmap="gray_r")
    plt.title(f"Class {Ytrn[1]}")
    plt.savefig("results/2_1_2.png")
    plt.show()
iaml212cw2_q2_1()

# Q2.2
def iaml212cw2_q2_2():
    #Q2.2 a)
    #pairwise euclidean distances of training set
    print(euclidean_distances(Xtrn_m, Xtrn_m))
    print(euclidean_distances(Xtrn, Xtrn))

    # Q2.2 b)
    #mean values of training and testing set
    np.mean(Xtst, axis=0)
    np.mean(Xtrn, axis=0)

iaml212cw2_q2_2()   # comment this out when you run the function

# Q2.3
def iaml212cw2_q2_3():
    # Q2.3 a)
    # k=3
    classes = [0,5,8]
    classcentres = []
    km3 = KMeans(n_clusters = 3, random_state=0)
    for i in classes:
        km3.fit(Xtrn[Ytrn==i])
        centres = km3.cluster_centers_
        for ci in centres:
            classcentres.append(ci)

    fig, axs = plt.subplots(3, 3) #ploting 3 by 3 histogram
    axs = axs.ravel()

    for i in range(len(classcentres)):
        axs[i].imshow(classcentres[i].reshape((28,28)).T,cmap = "gray_r")

    for i in range(len(classes)):
        axs[3*i].set(ylabel=f"Class {classes[i]}")
        axs[6+i].set(xlabel=f"Cluster {i+1}")

    fig.suptitle('Images of Cluster Centres for K=3')
    plt.savefig("results/2_3_1.png")
    plt.show()

    # k=5
    classcentres = []
    km5 = KMeans(n_clusters = 5, random_state=0)
    for i in classes:
        km5.fit(Xtrn[Ytrn==i])
        centres = km5.cluster_centers_
        for ci in centres:
            classcentres.append(ci)

    fig, axs = plt.subplots(3, 5) #ploting 3 by 3 histogram
    axs = axs.ravel()

    for i in range(len(classcentres)):
        axs[i].imshow(classcentres[i].reshape((28,28)).T,cmap = "gray_r")

    for i in range(len(classes)):
        axs[5*i].set(ylabel=f"Class {classes[i]}")
    for i in range(5):
        axs[10+i].set(xlabel=f"Cluster {i+1}")

    fig.suptitle('Images of Cluster Centres for K=5')
    plt.savefig("results/2_3_2.png")
    plt.show()
iaml212cw2_q2_3()   # comment this out when you run the function

# Q2.4
# def iaml212cw2_q2_4():

#       desribed in Assignment_2.pdf

# iaml212cw2_q2_4()   # comment this out when you run the function

# Q2.5
def iaml212cw2_q2_5():
    # Q2.5 a)
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(Xtrn_m, Ytrn)
    print(f'Classification accuracy on training set: {lr.score(Xtrn_m, Ytrn):.4f}')
    print(f'Classification accuracy on testing set: {lr.score(Xtst_m, Ytst):.4f}')

    # Q2.5 b)
    print(Ytst)
    print(lr.predict(Xtst_m))

    predicty = lr.predict(Xtst_m)
    nomatchidx = []
    for i in range(len(Ytst)):
        if (Ytst[i]!=predicty[i]):
            nomatchidx.append(Ytst[i])

    u, count = np.unique(nomatchidx, return_counts=True)
    countsort = np.argsort(-count)
    u = u[countsort]
    alphabet = []
    for i in u[0:5]:
        alphabet.append(chr(ord('@')+i+1))
    print(count[countsort][0:5])
    print(u[0:5])
    print(alphabet)
iaml212cw2_q2_5()   # comment this out when you run the function



# Q2.6
def iaml212cw2_q2_6():
    # Q2.6 c)

    # grid searching key hyperparametres for logistic regression
    # define models and parameters
    model = LogisticRegression(max_iter = 1000, random_state=0)
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['none', 'l1', 'l2', 'elasticnet']
    cvalues = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=cvalues)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    gridsearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy')
    gridresult = gridsearch.fit(Xtrn_m, Ytrn)
    # summarize results
    print(f"Best Accuracy: {gridresult.best_score_} using {gridresult.best_params_}")
    means = gridresult.cv_results_['mean_test_score']
    stds = gridresult.cv_results_['std_test_score']
    params = gridresult.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}" % (mean, stdev, param))

    #new classification accuracy on testing set
    lr = LogisticRegression(max_iter=1000, C=0.1, penalty='l2', solver='newton-cg', random_state=0)
    lr.fit(Xtrn_m, Ytrn)
    print(f'Classification accuracy on training set: {lr.score(Xtrn_m, Ytrn):.4f}')
    print(f'Classification accuracy on testing set: {lr.score(Xtst_m, Ytst):.4f}')
iaml212cw2_q2_6()   # comment this out when you run the function

# Q2.7
def iaml212cw2_q2_7():
    # Q2.7 a)
    covMatrix = np.cov(Xtrn_m[Ytrn==0], ddof=1)
    print(np.mean(covMatrix))
    print(np.max(covMatrix), np.min(covMatrix))

    # Q2.7 b)
    di = np.diag(covMatrix)
    print(np.mean(di))
    print(np.max(di), np.min(di))

    # Q2.7 c)
    plt.hist(di, bins=15, label="diagonal values")
    plt.title("Histogram of the diagonal values")
    plt.xlabel("Covariance values")
    plt.ylabel("Count")
    plt.grid()
    plt.legend()
    plt.savefig("results/2_7.png")
    plt.show()

    # Q2.7 d)
    meanvec = np.mean(Xtrn_m[Ytrn==0].T, axis=1)
    print(covMatrix)
    likelihood = scipy.stats.multivariate_normal.pdf(Xtst_m[Ytst==0][0], meanvec, covMatrix)
    # likelihood = rv.pdf(Xtrn_m[Ytrn==0][0])
iaml212cw2_q2_7()   # comment this out when you run the function

# Q2.8
def iaml212cw2_q2_8():
    # Q2.8 a)
    classAtrn = Xtrn_m[Ytrn==0]
    classAtst = Xtst_m[0,:]

    gmm = GaussianMixture(n_components=1, covariance_type='full').fit(classAtrn)
    print(f"The log likelihood is: {gmm.score(classAtst.reshape(1,784))}")

    # Q2.8 b)
    trainaccuracies = []
    testaccuracies = []
    for i in range(26):
        gmm = GaussianMixture(n_components=1, covariance_type='full').fit(Xtrn_m[Ytrn==i])
        ypred = gmm.predict(Xtrn_m)
        print(np.sum(np.diag(confusion_matrix(Ytrn, ypred))))
        trainaccuracies.append(accuracy_score(Ytrn, ypred))

        ypred = gmm.predict(Xtst_m)
        print(np.sum(np.diag(confusion_matrix(Ytst, ypred))))
        testaccuracies.append(accuracy_score(Ytst, ypred))
    print(trainaccuracies)
    print(testaccuracies)
iaml212cw2_q2_8()   # comment this out when you run the function

# Q2.9
# def iaml212cw2_q2_9():

#       desribed in Assignment_2.pdf

# iaml212cw2_q2_9()   # comment this out when you run the function
