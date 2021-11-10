
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
import matplotlib.pyplot as plt
import seaborn as sns
from iaml_cw2_helpers import *
# from iaml212cw2_my_helpers import *
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
    print('Number of instances: {}, number of attributes: {}'.format(Xtrn.shape[0], Xtrn.shape[1]))
    Xtrndf = pd.DataFrame(Xtrn)
    print(np.max(Xtrn), np.min(Xtrn), np.mean(Xtrn), np.std(Xtrn))
    # print(Xtrndf.describe())



    print('Number of instances: {}, number of attributes: {}'.format(Xtst.shape[0], Xtst.shape[1]))
    Xtstdf = pd.DataFrame(Xtst)
    print(np.max(Xtst), np.min(Xtst), np.mean(Xtst), np.std(Xtst))
    # print(Xtstdf.describe())
    # Xtst.describe()
iaml212cw2_q2_1()   # comment this out when you run the function

# Q2.2
# def iaml212cw2_q2_2():
#
# iaml212cw2_q2_2()   # comment this out when you run the function

# Q2.3
# def iaml212cw2_q2_3():
#
# iaml212cw2_q2_3()   # comment this out when you run the function

# Q2.4
# def iaml212cw2_q2_4():
#
# iaml212cw2_q2_4()   # comment this out when you run the function

# Q2.5
# def iaml212cw2_q2_5():
#
# iaml212cw2_q2_5()   # comment this out when you run the function

# Q2.6
# def iaml212cw2_q2_6():
#
# iaml212cw2_q2_6()   # comment this out when you run the function

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
