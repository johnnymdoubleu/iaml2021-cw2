
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml212cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from iaml_cw2_helpers import *
from iaml212cw2_my_helpers import *


X, Y = load_Q1_dataset()

Xtrn = X[100:,:]; Ytrn = Y[100:] # training set
Xtst = X[0:100,:]; Ytst = Y[0:100] # testing set

scaler = StandardScaler().fit(Xtrn)
Xtrn_s = scaler.transform(Xtrn) #scaling training set
Xtst_s = scaler.transform(Xtst) #scaling testing set

#<----

# Q1.1
def iaml212cw2_q1_1():
    Xa = Xtrn[np.where(Ytrn==0)[0]] #instances of class 0
    Xb = Xtrn[np.where(Ytrn==1)[0]] #instances of class 1

    fig, axs = plt.subplots(3, 3, figsize=(10,9)) #ploting 3 by 3 histogram
    axs = axs.ravel()

    for i in range(9):
        axs[i].hist([Xa[:,i], Xb[:,i]], bins=15)
        axs[i].set(xlabel=f"A{i}")

    for ax in axs.flat:
#         ax.set(ylabel='frequency')
        ax.grid()
        # ax.label_outer()
#     fig.tight_layout()
    fig.subplots_adjust(bottom=0.01)   ##  Need to play with this number.

    fig.legend(labels=['class 0', 'class 1'], loc="right")

    fig.suptitle("Histogram of each attribute by class")
    fig.supylabel("Count")
    plt.savefig("results/1_1.png")
    plt.show()
iaml212cw2_q1_1()

# Q1.2
def iaml212cw2_q1_2():
    for i in range(9):
        print(np.corrcoef(Xtrn[:,i], Ytrn)[0][1])
iaml212cw2_q1_2()   # comment this out when you run the function

# Q1.3
# def iaml212cw2_q1_3():

        # desribed in Assignment_2.pdf

 # iaml212cw2_q1_3()   # comment this out when you run the function
#
def iaml212cw2_q1_4():
    samVar = np.zeros((9,2)) #initialising numpy array to store the unbiased smaple variance

    #computing unbiased sample variance of each attribute
    for i in range(9):
        vari = np.array([np.var(Xtrn[:,i], axis=0, ddof=1),str(i)])
        samVar[i]=vari
    samVar = samVar[samVar[:,0].argsort()[::-1]] #sorting in descending order
    print(f"The unbiased sample variance of each attribute are {samVar}")

    #Q1.4 a)
    sum = samVar.sum(axis=0)[0]
    print(f"The sum of all the variances is {sum}")

    #Q1.4 b) i) plot
    plt.bar(range(len(samVar[:,0])), samVar[:,0], label="Variance explained by each of the Attributes")
    plt.title("Varaince explained by each of the Attributes")
    plt.xlabel("Attributes")
    plt.ylabel("Explained Variance")
    plt.xticks(np.arange(9), samVar[:,1].astype(str))
    plt.legend()
    plt.grid()
    plt.savefig("results/1_4_1.png")
    plt.show()

    #Q1.4 b) ii)
    cumsamVar = np.cumsum(samVar[:,0]/sum)
    print(f"Cumulative explained variance ratio of each Attributes are\n{cumsamVar}")
    plt.plot(cumsamVar, label="cumulative explained variance ratio")
    plt.title("cumulative explained variance ratio")
    plt.xlabel("Number of Attributes")
    plt.ylabel("Cumulative Explained Variance")
    plt.ylim(0,1.0)
    plt.xticks(np.arange(9), [1,2,3,4,5,6,7,8,9])
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("results/1_4_2.png")
    plt.show()

iaml212cw2_q1_4()   # comment this out when you run the function

# # Q1.5
def iaml212cw2_q1_5():
    pca = PCA().fit(Xtrn) #fitting Training set to PCA()
    #Q1.5 a)
    print(f"The total amount of variance is {sum(pca.explained_variance_)}")

    #Q1.5 b) i)
    plt.bar(range(1,10), pca.explained_variance_, label="Explained Variance")
    plt.title("Varaince explained by each Principal Componenets")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance")
    plt.xticks(np.arange(1,10))
    plt.legend()
    plt.grid()
    plt.savefig("results/1_5_1.png")
    plt.show()

    #Q1.5 b) ii)
    print(f"Cumulative explained variance ratio is {np.cumsum(pca.explained_variance_ratio_)}")
    plt.plot(np.cumsum(pca.explained_variance_ratio_), label="cumulative explained variance ratio")
    plt.title("cumulative explained variance ratio")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.xticks(np.arange(9),[1,2,3,4,5,6,7,8,9])
    plt.ylim(0,1.0)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("results/1_5_2.png")
    plt.show()

    #Q1.5 c)
    x2d = pca.fit(Xtrn)
    compo = x2d.components_[0:2] #taking the first two componets of PCA
    colours = ['b','r']
    classes = [0,1]
    newx = np.dot(Xtrn, compo.T) #computing projection

    #plotting
    for colour, target_name in zip(colours, classes):
        plt.scatter(newx[Ytrn==target_name, 0], newx[Ytrn==target_name, 1], color=colour, alpha=.8, s=10,
                    lw=2, label=f"class {target_name}")
    plt.grid()
    plt.legend()
    plt.title("Mapping of training instances on principal components")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("results/1_5_3.png")
    plt.show()

    #Q1/5 d)
    for i in range(9):
        print(np.corrcoef(Xtrn[:,i], newx[:,0])[0][1], np.corrcoef(Xtrn[:,i], newx[:,1])[0][1])

iaml212cw2_q1_5()   # comment this out when you run the function



#Q1.6
def iaml212cw2_q1_6():

    #Q1.6 a)
    pca = PCA().fit(Xtrn_s)
    print(f"The total amount of variance is {sum(pca.explained_variance_)}")

    #Q1.6 b) i)
    plt.bar(range(1,10), pca.explained_variance_, label="Explained Variance")
    plt.title("Varaince explained by each Principal Componenets")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance")
    plt.xticks(np.arange(1,10))
    plt.legend()
    plt.grid()
    plt.savefig("results/1_6_1.png")
    plt.show()

    #Q1.6 b) ii)
    print(f"cumsumVar is {np.cumsum(pca.explained_variance_ratio_)}")
    plt.plot(np.cumsum(pca.explained_variance_ratio_), label="cumulative explained variance ratio")
    plt.title("cumulative explained variance ratio")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.xticks(np.arange(9), [1,2,3,4,5,6,7,8,9])
    plt.legend()
    plt.ylim(0,1)
    plt.grid()
    plt.savefig("results/1_6_2.png")
    plt.show()

    #Q1.6 c)

    # Xtrn_sm = Xtrn_s - Xtrn_s.mean(axis=0)
    x2d = pca.fit(Xtrn_s)
    print(f"")
    compo = x2d.components_[0:2] #PCA Components
    colours = ['b','r']
    newx_s = np.dot(Xtrn_s, compo.T)
    classes = [0,1]
    for colour, target_name in zip(colours, classes):
        plt.scatter(newx_s[Ytrn==target_name, 0], newx_s[Ytrn==target_name, 1], c = colour, alpha=.8, lw=2,
                    label=f"class {target_name}", s=10)
    plt.title("Mapping of scaled training instances on principal components")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid()
    plt.savefig("results/1_6_3.png")
    plt.show()


    #Q1.6 d)
    for i in range(9):
        print(np.corrcoef(Xtrn_s[:,i], newx_s[:,0])[0][1], np.corrcoef(Xtrn_s[:,i], newx_s[:,1])[0][1])

iaml212cw2_q1_6()   # comment this out when you run the function

# Q1.7
# def iaml212cw2_q1_7():

#       desribed in Assignment_2.pdf

# iaml212cw2_q1_7()   # comment this out when you run the function
#
# Q1.8
def iaml212cw2_q1_8():
    #generating penalty parameters
    grid = {"C":np.logspace(-2,2,13)}
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    result = GridSearchCV(SVC(), grid, cv=cv, return_train_score= True)
    result.fit(Xtrn_s, Ytrn)


    means_test = result.cv_results_['mean_test_score']
    stds_test = result.cv_results_['std_test_score']
    means_train = result.cv_results_['mean_train_score']
    stds_train = result.cv_results_['std_train_score']
    params = result.cv_results_['params']
    print('mean, std for training set')
    #
    for mean, stdev, param in zip(means_train, stds_train, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print('mean, std for validation set')
    for mean, stdev, param in zip(means_test, stds_test, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # Q1.8 a)
    plt.errorbar(np.logspace(-2,2,13), means_train, stds_train, label="training set", capsize=5, fmt ='o-')
    plt.errorbar(np.logspace(-2,2,13), means_test, stds_test, label="validation set", capsize=5, fmt ='o-')
    plt.xscale('log')
    plt.title("Mean classification accuracy vs regularisation penalty parameters")
    plt.xlabel("Penalties")
    plt.ylabel("Mean accuracy")
    plt.grid()
    plt.legend()
    plt.savefig("results/1_8.png")
    plt.show()

    # Q1.8 c)
    print("tuned hpyerparameters :(best parameters) ",result.best_params_)
    print("accuracy :",result.best_score_)

    #Q1.8 d)
    svm = SVC(kernel="rbf", C = result.best_params_["C"])
    model = svm.fit(Xtrn_s, Ytrn)
    modelpre = model.predict(Xtst_s)
    cm = confusion_matrix(Ytst, modelpre)
    print(f"The classification accuracy is {model.score(Xtst_s, Ytst)}")
    print(f"number of instances correctly classified : {cm[0,0]+cm[1,1]}")
iaml212cw2_q1_8()   # comment this out when you run the function
#
# Q1.9
def iaml212cw2_q1_9():
    xxtrn = Xtrn[Ytrn==0]
    Ztrn = xxtrn[xxtrn[:,4]>=1]
    Ztrn = Ztrn[:,[4,7]]

    # Q1.9 a)
    #computing the mean vector and covariance matrix
    mean = Ztrn.mean(axis=0)
    covmat = np.cov(Ztrn[:,0], Ztrn[:,1],ddof=1)
    print(mean)
    print(covmat)

    #Q1.9 b)
    minx = np.min(Ztrn[:,0])
    maxx = np.max(Ztrn[:,0])
    miny = np.min(Ztrn[:,1])
    maxy = np.max(Ztrn[:,1])

    n = 318
    x = np.linspace(miny, maxx, n)
    y = np.linspace(miny, maxx, n)

    x,y = np.meshgrid(x,y)
    pos = np.dstack((x,y))
    rv = scipy.stats.multivariate_normal(mean, covmat)
    z = rv.pdf(pos)
    plt.figure(figsize=(9,6))
    plt.contour(x,y,z, cmap="terrain")
    plt.scatter(Ztrn[:,0], Ztrn[:,1], color='r', label="instances")

    plt.xlabel('A4')
    plt.ylabel('A7')
    plt.title("Scatter and Contour Plot of Estimated Gaussian Distribution")
    plt.grid()
    plt.legend()
    plt.savefig("results/1_9.png")
    plt.show()
iaml212cw2_q1_9()   # comment this out when you run the function
#
# Q1.10
def iaml212cw2_q1_10():
    xxtrn = Xtrn[Ytrn==0]
    Ztrn = xxtrn[xxtrn[:,4]>=1]
    Ztrn = Ztrn[:,[4,7]]

    gnb = GaussianNB()
    gnb.fit(Ztrn, np.zeros(318))
    print(gnb.theta_[0])
    meannb = gnb.theta_[0]
    print(f"The mean vector is: {meannb}")
    # ztrn_cov_nb = np.diag(gnb.sigma_[0]*(len(Ztrn)-1)/len(Ztrn))
    covnb = np.diag(gnb.sigma_[0])
    print(f"The covariance matrix is:\n{covnb}")

    minx = np.min(Ztrn[:,0])
    maxx = np.max(Ztrn[:,0])
    miny = np.min(Ztrn[:,1])
    maxy = np.max(Ztrn[:,1])

    n = 318
    x = np.linspace(miny, maxx, n)
    y = np.linspace(miny, maxx, n)
    x,y = np.meshgrid(x,y)
    pos = np.dstack((x,y))
    rv = scipy.stats.multivariate_normal(meannb, covnb)
    z = rv.pdf(pos)
    plt.figure(figsize=(9,6))
    plt.contour(x, y, z, cmap="terrain")
    plt.scatter(Ztrn[:,0], Ztrn[:,1], color='r', label='instances')
    plt.xlabel('A4')
    plt.ylabel('A7')
    plt.title("Scatter and Contour Plot of Estimated Gaussian Distribution with Naive Bayes")
    plt.grid()
    plt.legend()
    plt.savefig("results/1_10.png")
    plt.show()
iaml212cw2_q1_10()   # comment this out when you run the function
