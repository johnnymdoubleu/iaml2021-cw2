
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from iaml_cw2_helpers import *
from iaml212cw2_my_helpers import *


X, Y = load_Q1_dataset()

Xtrn = X[100:,:]; Ytrn = Y[100:] # training set
# print(Xtrn.shape, Ytrn.shape)
Xtst = X[0:100,:]; Ytst = Y[0:100] # testing set
#<----

# Q1.1
def iaml212cw2_q1_1():
    Xa = Xtrn[np.where(Ytrn==0)[0]] #instances of class 0
    Xb = Xtrn[np.where(Ytrn==1)[0]] #instances of class 1

    fig, axs = plt.subplots(3, 3) #ploting 3 by 3 histogram
    axs = axs.ravel()

    for i in range(9):
        axs[i].hist([Xa[:,i], Xb[:,i]], bins=15)
        axs[i].set(xlabel=f"A{i}")

    for ax in axs.flat:
        ax.set(ylabel='frequency')
        ax.grid()
        # ax.label_outer()
    plt.savefig("results/1_1.png")
    plt.show()
# iaml212cw2_q1_1()   # comment this out when you run the function

# Q1.2
def iaml212cw2_q1_2():
    for i in range(9):
        print(np.corrcoef(Xtrn[:,i], Ytrn))
# iaml212cw2_q1_2()   # comment this out when you run the function
#
# [[1.        0.4911759]
#  [0.4911759 1.       ]]
# [[1.        0.0874059]
#  [0.0874059 1.       ]]
# [[1.         0.22728719]
#  [0.22728719 1.        ]]
# [[1.         0.20736605]
#  [0.20736605 1.        ]]
# [[1.         0.10772035]
#  [0.10772035 1.        ]]
# [[1.        0.1856714]
#  [0.1856714 1.       ]]
# [[1.         0.07626074]
#  [0.07626074 1.        ]]
# [[1.         0.30445377]
#  [0.30445377 1.        ]]
# [[1.         0.24034733]
#  [0.24034733 1.        ]]

# Q1.3
# def iaml212cw2_q1_3():

        # desribed in Assignment_2.pdf

 # iaml212cw2_q1_3()   # comment this out when you run the function
#
# Q1.4
def iaml212cw2_q1_4():
    samVar = np.zeros((9,2)) #initialising numpy array to store the unbiased smaple variance
    # print(samVar)
    for i in range(9):
        #computing unbiased sample variance of each attribute
        vari = np.array([np.var(Xtrn[:,i], axis=0, ddof=1),str(i)])
        samVar[i]=vari
    samVar = samVar[samVar[:,0].argsort()[::-1]] #sorting in descending order
#     print(samVar)
    print(f"The unbiased sample variance of each attribute are {samVar}")
    # print(samVar[:,0])

    #Q1.4.1 plot
    plt.plot(samVar[:,0], label="Varaince explained by each of the Attributes")
    plt.title("Varaince explained by each of the Attributes")
    plt.xlabel("Attributes")
    plt.ylabel("Explained Variance")
    plt.xticks(np.arange(9), samVar[:,1].astype(str))
    plt.legend()
    plt.grid()
    plt.savefig("results/1_4_1.png")
    plt.show()

    sum = samVar.sum(axis=0)[0]
    print(f"The sum of all the variances is {sum}")
    cumsamVar = np.cumsum(samVar[:,0]/sum)
    print(f"Cumulative explained variance ratio of each Attributes are {cumsamVar}")
    #Q1.4.2 plot
    plt.plot(cumsamVar, label="cumulative explained variance ratio")
    plt.title("cumulative explained variance ratio")
    plt.xlabel("Attributes")
    plt.ylabel("Cumulative Explained Variance")
    plt.xticks(np.arange(9), samVar[:,1].astype(str))
    plt.legend()
    plt.grid()
    plt.savefig("results/1_4_2.png")
    plt.show()

# iaml212cw2_q1_4()   # comment this out when you run the function

# # Q1.5
def iaml212cw2_q1_5():
    pca = PCA().fit(Xtrn)
    print(f"The total amount of variance is {sum(pca.explained_variance_)}")

    plt.plot(pca.explained_variance_, label="Varaince explained by each Principal Components")
    plt.title("Varaince explained by each Principal Componenets")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance")
    # plt.xticks(np.arange(9), samVar[:,1].astype(str))
    plt.legend()
    plt.grid()
    plt.savefig("results/1_5_1.png")
    plt.show()

    print(f"Cumulative explained variance ratio is {np.cumsum(pca.explained_variance_ratio_)}")
    plt.plot(np.cumsum(pca.explained_variance_ratio_), label="cumulative explained variance ratio")
    plt.title("cumulative explained variance ratio")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    # plt.xticks(np.arange(9), samVar[:,1].astype(str))
    plt.legend()
    plt.grid()
    plt.savefig("results/1_5_2.png")
    plt.show()

    pca = PCA(2)
    Xtrn_m = Xtrn - Xtrn.mean(axis=0)
    x2d = pca.fit(Xtrn_m)
    compo = x2d.components_
    colours = ['b','r']

    classes = [0,1]
    newx = np.dot(Xtrn_m, compo.T)
    for colour, target_name in zip(colours, classes):
        plt.scatter(newx[Ytrn==target_name, 0], newx[Ytrn==target_name, 1], color=colour, alpha=.8, s=10, lw=2)

    plt.grid()
    plt.title("Principal Component of Training Set")
    plt.show()

    print(np.corrcoef(Xtrn, compo[0]))
    print(np.corrcoef(Xtrn, compo[1]))
# iaml212cw2_q1_5()   # comment this out when you run the function   # comment this out when you run the function

#Q1.6
def iaml212cw2_q1_6():
    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)

    Xtrn_sm = Xtrn_s - Xtrn_s.mean(axis=0)
    pca = PCA(2)
    x2d = pca.fit(Xtrn_sm)
    compo = x2d.components_
    newx = np.dot(Xtrn_sm, compo.T)
    colours = ['b','r']
    classes = [0,1]
    for colour, target_name in zip(colours, classes):
        plt.scatter(newx[Ytrn==target_name, 0], newx[Ytrn==target_name, 1], c = colour, alpha=.8, lw=2,
                    label=target_name, s=10)
    plt.grid()
    plt.show()

iaml212cw2_q1_6()   # comment this out when you run the function
#
# # Q1.7
# def iaml212cw2_q1_7():
# #
# # iaml212cw2_q1_7()   # comment this out when you run the function
#
# # Q1.8
# def iaml212cw2_q1_8():
# #
# # iaml212cw2_q1_8()   # comment this out when you run the function
#
# # Q1.9
# def iaml212cw2_q1_9():
# #
# # iaml212cw2_q1_9()   # comment this out when you run the function
#
# # Q1.10
# def iaml212cw2_q1_10():
# #
# # iaml212cw2_q1_10()   # comment this out when you run the function
