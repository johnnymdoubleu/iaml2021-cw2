
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
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pandas.api.types import CategoricalDtype
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
    print(f'Number of instances: {Xtrn.shape[0]}, number of attributes: {Xtrn.shape[1]}')
    Xtrndf = pd.DataFrame(Xtrn)
    print(np.max(Xtrn), np.min(Xtrn), np.mean(Xtrn), np.std(Xtrn))
    # print(Xtrndf.describe())

    print(f'Number of instances: {Xtst.shape[0]}, number of attributes: {Xtst.shape[1]}')
    Xtstdf = pd.DataFrame(Xtst)
    print(np.max(Xtst), np.min(Xtst), np.mean(Xtst), np.std(Xtst))
    # print(Xtstdf.describe())
    # Xtst.describe()
    plt.imshow(Xtrn[0].reshape((28,28)), cmap="gray_r")
    plt.title(f"Class {Ytrn[0]}")
    plt.show()
    plt.imshow(Xtrn[1].reshape((28,28)), cmap="gray_r")
    plt.title(f"Class {Ytrn[1]}")
    plt.show()
iaml212cw2_q2_1()

# Q2.2
def iaml212cw2_q2_2():
    euclidean_distances(Xtrn_m, Xtrn_m)
    euclidean_distances(Xtrn, Xtrn)
# they are exactly the same. Normalised means both instances are mean vector subtracted.
#Thus subtracting two instances will cancel of the mean and eventually compute the same distance
    np.mean(Xtst, axis=0)
    np.mean(Xtrn, axis=0)
# We can clearly see the difference in the mean vector.
# However, we want only want to use the mean vector of the training set.
# This is because we want to keep the testing data to be a new set of data
# this means that we are testing the test set with prior knowledge of the data set leading inaccuracy
# sampling errors may negatively bias the predictions
# our aim is to test and evaluate whether our model can fit the testing data.

# iaml212cw2_q2_2()   # comment this out when you run the function

# Q2.3
def iaml212cw2_q2_3():
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
        axs[i].imshow(classcentres[i].reshape((28,28)),cmap = "gray_r")
        axs[i].set(xlabel=f"A{i}")

    plt.savefig("results/2_3_1.png")
    plt.show()

    classes = [0,5,8]
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
        axs[i].imshow(classcentres[i].reshape((28,28)),cmap = "gray_r")
        axs[i].set(xlabel=f"A{i}")

    plt.savefig("results/2_3_2.png")
    plt.show()
# iaml212cw2_q2_3()   # comment this out when you run the function

# Q2.4
# def iaml212cw2_q2_4():
#
# iaml212cw2_q2_4()   # comment this out when you run the function

# Q2.5
def iaml212cw2_q2_5():
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(Xtrn_m, Ytrn)
    print(f'Classification accuracy on training set: {lr.score(Xtrn_m, Ytrn):.3f}')
    print(f'Classification accuracy on testing set: {lr.score(Xtst_m, Ytst):.3f}')

    print(lr.predict(Xtst_m))
    print(Ytst)

    predicty = lr.predict(Xtst_m)
    nomatchidx = []
    for i in range(len(Ytst)):
        if (Ytst[i]!=predicty[i]):
            nomatchidx.append(Ytst[i])
    np.unique(nomatchidx, return_counts=True)

    u, count = np.unique(nomatchidx, return_counts=True)
    countsort = np.argsort(-count)
    u = u[countsort]
    u[0:5]
    alphabet = []
    for i in u[0:5]:
        alphabet.append(chr(ord('@')+i+1))
    print(u[0:5])
    print(alphabet)

# iaml212cw2_q2_5()   # comment this out when you run the function

# Q2.6
def iaml212cw2_q2_6():
    lr = LogisticRegression(max_iter=10000, random_state=0)
    lr.fit(Xtrn_m, Ytrn)
    print(f'Classification accuracy on training set: {lr.score(Xtrn_m, Ytrn):.3f}')
    print(f'Classification accuracy on testing set: {lr.score(Xtst_m, Ytst):.3f}')
iaml212cw2_q2_6()   # comment this out when you run the function

# Q2.7
# def iaml212cw2_q2_7():
#
# iaml212cw2_q2_7()   # comment this out when you run the function

# Q2.8
# def iaml212cw2_q2_8():
#
# iaml212cw2_q2_8()   # comment this out when you run the function

# Q2.9
# def iaml212cw2_q2_9():
#
# iaml212cw2_q2_9()   # comment this out when you run the function
