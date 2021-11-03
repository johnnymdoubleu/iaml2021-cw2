
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
from iaml_cw2_helpers import *
from iaml212cw2_my_helpers import *

print_versions()
X, Y = load_Q1_dataset()
# print(f"X: {X.shape}, Y: {Y.shape}")
Xtrn = X[100:,:]; Ytrn = Y[100:] # training set
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

    plt.show()
iaml212cw2_q1_1()   # comment this out when you run the function

# Q1.2
# def iaml212cw2_q1_2():
#     for i in range(9):

# iaml212cw2_q1_2()   # comment this out when you run the function
#
# # Q1.3
# def iaml212cw2_q1_3():
# #
# # iaml212cw2_q1_3()   # comment this out when you run the function
#
# # Q1.4
# def iaml212cw2_q1_4():
# #
# # iaml212cw2_q1_4()   # comment this out when you run the function
#
# # Q1.5
# def iaml212cw2_q1_5():
# #
# # iaml212cw2_q1_5()   # comment this out when you run the function
#
# # Q1.6
# def iaml212cw2_q1_6():
# #
# # iaml212cw2_q1_6()   # comment this out when you run the function
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
