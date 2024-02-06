#1
from sklearn.datasets import load_diabetes
data = load_diabetes()
X, y = data.data, data.target
print(data.DESCR)
#describe data
'''
Topic of the data:
The topic of the data is diabetes

Size of the data
The size of the data is 4,862 this comes from the 10 attribute columns and 1 target column so 11 columns *442 instances

Which is the target feature
The target feature is a quantitative measure of the diabetes progression one year after the baseline

What kind of plots you want to include in the EDA
I would want to do a scatterplot matrix to see if there is any colinearity in the response variables. I would also normally want to see the distrubution of each of the response datasets but in the note they say have been mean centered and scaled by standard devation meaning they are normally distrubuted. But I would want to do a scatterplot of the target variables to see the distrubution

Describe three steps that you think are necessary to pre-process the data
I would want to check for any na values
Check for any outliers
split into test and train dataset

'''
#2
'''
A)
matrix a=   [1,2,3,4]
            [5,6,7,8]
matrix b=   [1,2]
            [3,4]
            [5,6]
            [7,8]
B)
We cannot add theses matricies because they do not have the same dimmensions
C)
Yes we can because one we transpose a it turns into a 2*4 matrix and then the matricies will have the same dimmensions
D)
AB
will be a 2*2 matrix by 
[1*1+2*3+3*5+4*7, 1*2+2*4+3*6+4*8]
[5*1+6*3+7*5+8*7, 5*2+6*4+7*6+8*8]
[50, 60]
[114, 140]
matrix BA will be a 4*4 matrix that is 
[1*1+5*2, 2*1+6*2, 3*1+7*2, 4*1+8*2]
[1*3+5*4, 2*3+6*4, 3*3+7*4, 4*3+8*4]
[1*5+5*6, 2*5+6*6, 3*5+7*6, 4*5+8*6]
[1*7+5*8, 2*7+6*8, 3*7+7*8, 4*7+8*8]
which turns into
[11, 14, 17, 20]
[23, 30, 37, 44]
[35, 46, 57, 68]
[47, 62, 77, 92]
E)
No we cannot caluculate A inverse because it is not square matrix.
you can take the inverse of ab by doing 1/determinant [d, -b][-c, a]
determinant is ad - bc = 7000-6840=160
[7/8, -3/8]
[-57/80, 5/18]
'''
#matrix in coding
import numpy as np
arraya = np.random.randint(10, size=(2, 4))
arrayb = np.random.randint(10, size=(4, 2))
print(arraya)
print(arrayb)
try:
    print(np.add(arraya,arrayb))
except ValueError:
    print('This problem cannot be done because the arrays are of different dimmensions' )
print(np.add(arraya.transpose(),arrayb))
print(np.matmul(arraya, arrayb))
print(np.matmul(arraya, arrayb))
try:
    print(np.linalg.inv(arraya))
except:
    print('This problem cannot be done because the arrays are of different dimmensions' )
try:
    print(np.linalg.inv(np.matmul(arraya, arrayb)))
except:
    print('This problem cannot be done because the arrays are of different dimmensions' )
#3
#a) 
#.6*.6=.36
#b)
#.4*.4=.16
#c)
#.4*.6+.6*.4=.48


#4
#a)
#.6*.19+.1*.73+.3*.4
#=0.307
#b)
#0.073 ÷ (0.073 + 0.12 + 0.114)
#=0.238
#0.114 ÷ (0.073 + 0.12 + 0.114)
#=0.371
