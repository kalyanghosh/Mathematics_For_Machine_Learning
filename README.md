

<b>MATHEMATICS FOR MACHINE LEARNING</b>
<br></br>
This repository contains code and theory of mathematical concepts required to master Machine Learning.
This repository will give any budding beginner in Machine Learning a solid foundation on the important concepts 
in <b>Linear Algebra</b> and <b>Multivariate Calculus</b>.I have included IPython notebooks which has code snippets and important
explanations.
<br></br>
<b>So, Go ahead, Fork this repo and Happy Machine Learning :)</b>
<br></br>
<b>TABLE OF CONTENTS: </b>
<br></br>
<b>1. LINEAR ALGEBRA: <b>
<br></br>

-Linear Algebra Intuition:
```python
#Definition of Linear Algebra:

#Ans: Linear Algebra can be defined as the study of vectors,vector spaces and mapping between vector spaces.
#     It emerged from the study of matrices and the knowledge that they can be solved using system of linear equations.

#Use cases of Linear Algebra:

#Ans: 1. Solving of linear equations like:
         2a+3b=10
         7b+5c=20
         The above eqaution can be decomposed into matrices and vectors like:
         [2 3 [a]=[10]
          7 5][b] [20]
#     2. Optimization Problem:
#        Another use can be to fit a line that best matches a data distribution.The objective is to find the 
#        line or curve that best fits the data distribution by finding the optimal parameters of the line.

# Exercise:

#Solve the system of equations given by:

3x−2y+z=7
x+y+z=2
3x−2y−z=3
```
<br></br>

-Vector Space in DataScience:
```python
#Vectors in Machine Learning:

Ans: From a Computer Science point of view,A vector can be defined as a list of numbers.
     From a Physical point of view, a vector can be defined as a position in 3D space.
     If we visualize the parameters or features of an entity, a vector can be visualized spatially as
     a point in the 3D space.The vector can move in the the N dimensional feature space to find the globally
     locally optimum set of features  in the feature space.
     So, it is very important to visulize the features as vectors in the N dimensional feature space, so Linear 
     Algebra routines, and calculus formulaes can be applied in oder to solve them.
```
<br></br>
-Vector_Operations:
```python
#Vector Operations:

# There are two fundamental operations that can be performed on vectors

# 1. Vector addition and Subtraction:
#    Lets us define a co-ordinate space X,Y with two unit vectors X and Y. The space spanned by the 
#    unit vectors is called the co-ordinate space.These unit vectors X and Y are called the basis vectors.

#    Vector addition and subtraction follows the associativity rule:
#    A+(B+C)=(A+B)+C

# 2. Multiplication by a scalar constant
#    If we multiply a vector with a scalar constant, each of the components of the scalar vector is multiplied
#    by the scalar constant.
#    c[A] = [cA]
#     [B]   [cB]
```
<br></br>
-Cosine_Dot_Product:
```python
#Modulus  and dot product of vectors
 # 1. The dot product or the projection product of two vectors can be explained as :
      r.s = |r||s|cos(theta) , where theta is the angle between the two vectors
      Dot product of two vectors satisfy the following properties:
      * r.s = s.r (Commutative)
      * r.(s+t) = r.s+r.t (Distributive)
      * r.(ks) = k(rs) (Associative over Scalar Multiplication)
 #2. If two vectors are orthogonal to each other , the the value of :
      * r.s = 0 , because the value of cos(90)=0
     If two vectors are parallel to each other, then the value of:
      * r.s = |r||s|, because the value of cos(0)=1
```
<br></br>
-Vector_Projection:
```python
# Projection:
  The dot product of two vectors is defined as :
  * a.b=|a||b|cos(theta)
  * The term |b|cos(theta) is defined as the projection of the vector b on to a
    Now, if we write :
    > (a.b)/|a|=|b|cos(theta), is called the scalar projection
    >  a(a.b)/|a||a|= vector projection
```
<br></br>
-Matrix_Inverse:
```python
#Performing matrix inverse
import numpy as np

A=[[1,1,3],
  [1,2,4],
  [1,1,2]]
Ainv=np.linalg.inv(A)
print ("The inverse of matrix ",A," is = ",Ainv)
```
```python
#Solving the system of linear equations instead of taking inverse which is computationally expensive
import numpy as np
A = [[4, 6, 2],
     [3, 4, 1],
     [2, 8, 13]]
s = [9, 7, 2]
r=np.linalg.solve(A,s)
print ("The value of vector r in Ar=s is = ",r )
```
```python
import numpy as np
A=[[1,1,1],
   [3,2,1],
   [2,1,2]]
Ainv =np.linalg.inv(A)
print ("The value of inverse = ",Ainv)
```
<br></br>
-Coding Examples:
```python
'''
Identifying special matrices
Instructions
In this assignment, you shall write a function that will test if a 4×4 matrix is singular,
i.e. to determine if an inverse exists, before calculating it.

You shall use the method of converting a matrix to echelon form, and testing if this fails by 
leaving zeros that can’t be removed on the leading diagonal.


Matrices in Python
In the numpy package in Python, matrices are indexed using zero for the top-most column and left-most row. 
I.e., the matrix structure looks like this:

A[0, 0]  A[0, 1]  A[0, 2]  A[0, 3]
A[1, 0]  A[1, 1]  A[1, 2]  A[1, 3]
A[2, 0]  A[2, 1]  A[2, 2]  A[2, 3]
A[3, 0]  A[3, 1]  A[3, 2]  A[3, 3]
You can access the value of each element individually using,

A[n, m]
which will give the n'th row and m'th column (starting with zero). You can also access a whole row at a time using,

A[n]
'''

import numpy as np

# Our function will go through the matrix replacing each row in order turning it into echelon form.
# If at any point it fails because it can't put a 1 in the leading diagonal,
# we will return the value True, otherwise, we will return False.
# There is no need to edit this function.
def isSingular(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False

# This next line defines our error flag. For when things go wrong if the matrix is singular.
# There is no need to edit this line.
class MatrixIsSingular(Exception): pass

# For Row Zero, all we require is the first element is equal to 1.
# We'll divide the row by the value of A[0, 0].
# This will get us in trouble though if A[0, 0] equals 0, so first we'll test for that,
# and if this is true, we'll add one of the lower rows to the first one before the division.
# We'll repeat the test going down each lower row until we can do the division.
# There is no need to edit this function.
def fixRowZero(A) :
    if A[0,0] == 0 :
        A[0] = A[0] + A[1]
    if A[0,0] == 0 :
        A[0] = A[0] + A[2]
    if A[0,0] == 0 :
        A[0] = A[0] + A[3]
    if A[0,0] == 0 :
        raise MatrixIsSingular()
    A[0] = A[0] / A[0,0]
    return A

# First we'll set the sub-diagonal elements to zero, i.e. A[1,0].
# Next we want the diagonal element to be equal to one.
# We'll divide the row by the value of A[1, 1].
# Again, we need to test if this is zero.
# If so, we'll add a lower row and repeat setting the sub-diagonal elements to zero.
# There is no need to edit this function.
def fixRowOne(A) :
    A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[2]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[3]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        raise MatrixIsSingular()
    A[1] = A[1] / A[1,1]
    return A

# This is the first function that you should complete.
# Follow the instructions inside the function at each comment.
def fixRowTwo(A) :
    # Insert code below to set the sub-diagonal elements of row two to zero (there are two of them).
    A[2]=A[2]-A[0]*A[2,0]
    A[2]=A[2]-A[1]*A[2,1]
    
    
    # Next we'll test that the diagonal element is not zero.
    if A[2,2] == 0 :
        # Insert code below that adds a lower row to row 2.
        A[2]=A[2]+A[3]
        
        # Now repeat your code which sets the sub-diagonal elements to zero.
        A[2]=A[2]-A[0]*A[2,0]
        A[2]=A[2]-A[1]*A[2,1]
    if A[2,2] == 0 :
        raise MatrixIsSingular()
    # Finally set the diagonal element to one by dividing the whole row by that element.
    A[2]=A[2]/A[2,2]
    return A

# You should also complete this function
# Follow the instructions inside the function at each comment.
def fixRowThree(A) :
    # Insert code below to set the sub-diagonal elements of row three to zero.
    A[3]=A[3]-A[0]*A[3,0]
    A[3]=A[3]-A[1]*A[3,1]
    A[3]=A[3]-A[2]*A[3,2]
    
    # Complete the if statement to test if the diagonal element is zero.
    if A[3,3]==0:
        raise MatrixIsSingular()
    # Transform the row to set the diagonal element to one.
    A[3]=A[3]/A[3,3]
    return A
```
