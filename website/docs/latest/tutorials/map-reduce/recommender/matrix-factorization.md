<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
---
layout: doc-page
title: (Deprecated)  Perceptron and Winnow

    
---
<a name="MatrixFactorization-Intro"></a>
# Introduction to Matrix Factorization for Recommendation Mining

In the mathematical discipline of linear algebra, a matrix decomposition 
or matrix factorization is a dimensionality reduction technique that factorizes a matrix into a product of matrices, usually two. 
There are many different matrix decompositions, each finds use among a particular class of problems.

In mahout, the SVDRecommender provides an interface to build recommender based on matrix factorization.
The idea behind is to project the users and items onto a feature space and try to optimize U and M so that U \* (M^t) is as close to R as possible:

     U is n * p user feature matrix, 
     M is m * p item feature matrix, M^t is the conjugate transpose of M,
     R is n * m rating matrix,
     n is the number of users,
     m is the number of items,
     p is the number of features

We usually use RMSE to represent the deviations between predictions and atual ratings.
RMSE is defined as the squared root of the sum of squared errors at each known user item ratings.
So our matrix factorization target could be mathmatically defined as:

     find U and M, (U, M) = argmin(RMSE) = argmin(pow(SSE / K, 0.5))
     
     SSE = sum(e(u,i)^2)
     e(u,i) = r(u, i) - U[u,] * (M[i,]^t) = r(u,i) - sum(U[u,f] * M[i,f]), f = 0, 1, .. p - 1
     K is the number of known user item ratings.

<a name="MatrixFactorization-Factorizers"></a>

Mahout has implemented matrix factorization based on 

    (1) SGD(Stochastic Gradient Descent)
    (2) ALSWR(Alternating-Least-Squares with Weighted-λ-Regularization).

## SGD

Stochastic gradient descent is a gradient descent optimization method for minimizing an objective function that is written as a su of differentiable functions.

       Q(w) = sum(Q_i(w)), 

where w is the parameters to be estimated,
      Q(w) is the objective function that could be expressed as sum of differentiable functions,
      Q_i(w) is associated with the i-th observation in the data set 

In practice, w is estimated using an iterative method at each single sample until an approximate miminum is obtained,

      w = w - alpha * (d(Q_i(w))/dw),
where aplpha is the learning rate,
      (d(Q_i(w))/dw) is the first derivative of Q_i(w) on w.

In matrix factorization, the RatingSGDFactorizer class implements the SGD with w = (U, M) and objective function Q(w) = sum(Q(u,i)),

       Q(u,i) =  sum(e(u,i) * e(u,i)) / 2 + lambda * [(U[u,] * (U[u,]^t)) + (M[i,] * (M[i,]^t))] / 2

where Q(u, i) is the objecive function for user u and item i,
      e(u, i) is the error between predicted rating and actual rating,
      U[u,] is the feature vector of user u,
      M[i,] is the feature vector of item i,
      lambda is the regularization parameter to prevent overfitting.

The algorithm is sketched as follows:
  
      init U and M with randomized value between 0.0 and 1.0 with standard Gaussian distribution   
      
      for(iter = 0; iter < numIterations; iter++)
      {
          for(user u and item i with rating R[u,i])
          {
              predicted_rating = U[u,] *  M[i,]^t //dot product of feature vectors between user u and item i
              err = R[u, i] - predicted_rating
              //adjust U[u,] and M[i,]
              // p is the number of features
              for(f = 0; f < p; f++) {
                 NU[u,f] = U[u,f] - alpha * d(Q(u,i))/d(U[u,f]) //optimize U[u,f]
                         = U[u, f] + alpha * (e(u,i) * M[i,f] - lambda * U[u,f]) 
              }
              for(f = 0; f < p; f++) {
                 M[i,f] = M[i,f] - alpha * d(Q(u,i))/d(M[i,f])  //optimize M[i,f] 
                        = M[i,f] + alpha * (e(u,i) * U[u,f] - lambda * M[i,f]) 
              }
              U[u,] = NU[u,]
          }
      }

## SVD++

SVD++ is an enhancement of the SGD matrix factorization. 

It could be considered as an integration of latent factor model and neighborhood based model, considering not only how users rate, but also who has rated what. 

The complete model is a sum of 3 sub-models with complete prediction formula as follows: 
    
    pr(u,i) = b[u,i] + fm + nm   //user u and item i
    
    pr(u,i) is the predicted rating of user u on item i,
    b[u,i] = U + b(u) + b(i)
    fm = (q[i,]) * (p[u,] + pow(|N(u)|, -0.5) * sum(y[j,])),  j is an item in N(u)
    nm = pow(|R(i;u;k)|, -0.5) * sum((r[u,j0] - b[u,j0]) * w[i,j0]) + pow(|N(i;u;k)|, -0.5) * sum(c[i,j1]), j0 is an item in R(i;u;k), j1 is an item in N(i;u;k)

The associated regularized squared error function to be minimized is:

    {sum((r[u,i] - pr[u,i]) * (r[u,i] - pr[u,i]))  - lambda * (b(u) * b(u) + b(i) * b(i) + ||q[i,]||^2 + ||p[u,]||^2 + sum(||y[j,]||^2) + sum(w[i,j0] * w[i,j0]) + sum(c[i,j1] * c[i,j1]))}

b[u,i] is the baseline estimate of user u's predicted rating on item i. U is users' overall average rating and b(u) and b(i) indicate the observed deviations of user u and item i's ratings from average. 

The baseline estimate is to adjust for the user and item effects - i.e, systematic tendencies for some users to give higher ratings than others and tendencies
for some items to receive higher ratings than other items.

fm is the latent factor model to capture the interactions between user and item via a feature layer. q[i,] is the feature vector of item i, and the rest part of the formula represents user u with a user feature vector and a sum of features of items in N(u),
N(u) is the set of items that user u have expressed preference, y[j,] is feature vector of an item in N(u).

nm is an extension of the classic item-based neighborhood model. 
It captures not only the user's explicit ratings but also the user's implicit preferences. R(i;u;k) is the set of items that have got explicit rating from user u and only retain top k most similar items. r[u,j0] is the actual rating of user u on item j0, 
b[u,j0] is the corresponding baseline estimate.

The difference between r[u,j0] and b[u,j0] is weighted by a parameter w[i,j0], which could be thought as the similarity between item i and j0. 

N[i;u;k] is the top k most similar items that have got the user's preference.
c[i;j1] is the paramter to be estimated. 

The value of w[i,j0] and c[i,j1] could be treated as the significance of the 
user's explicit rating and implicit preference respectively.

The parameters b, y, q, w, c are to be determined by minimizing the the associated regularized squared error function through gradient descent. We loop over all known ratings and for a given training case r[u,i], we apply gradient descent on the error function and modify the parameters by moving in the opposite direction of the gradient.

For a complete analysis of the SVD++ algorithm,
please refer to the paper [Yehuda Koren: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model, KDD 2008](http://research.yahoo.com/files/kdd08koren.pdf).
 
In Mahout,SVDPlusPlusFactorizer class is a simplified implementation of the SVD++ algorithm.It mainly uses the latent factor model with item feature vector, user feature vector and user's preference, with pr(u,i) = fm = (q[i,]) \* (p[u,] + pow(|N(u)|, -0.5) * sum(y[j,])) and the parameters to be determined are q, p, y. 

The update to q, p, y in each gradient descent step is:

      err(u,i) = r[u,i] - pr[u,i]
      q[i,] = q[i,] + alpha * (err(u,i) * (p[u,] + pow(|N(u)|, -0.5) * sum(y[j,])) - lamda * q[i,]) 
      p[u,] = p[u,] + alpha * (err(u,i) * q[i,] - lambda * p[u,])
      for j that is an item in N(u):
         y[j,] = y[j,] + alpha * (err(u,i) * pow(|N(u)|, -0.5) * q[i,] - lambda * y[j,])

where alpha is the learning rate of gradient descent, N(u) is the items that user u has expressed preference.

## Parallel SGD

Mahout has a parallel SGD implementation in ParallelSGDFactorizer class. It shuffles the user ratings in every iteration and 
generates splits on the shuffled ratings. Each split is handled by a thread to update the user features and item features using 
vanilla SGD. 

The implementation could be traced back to a lock-free version of SGD based on paper 
[Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](http://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf).

## ALSWR

ALSWR is an iterative algorithm to solve the low rank factorization of user feature matrix U and item feature matrix M.  
The loss function to be minimized is formulated as the sum of squared errors plus [Tikhonov regularization](http://en.wikipedia.org/wiki/Tikhonov_regularization):

     L(R, U, M) = sum(pow((R[u,i] - U[u,]* (M[i,]^t)), 2)) + lambda * (sum(n(u) * ||U[u,]||^2) + sum(n(i) * ||M[i,]||^2))
 
At the beginning of the algorithm, M is initialized with the average item ratings as its first row and random numbers for the rest row.  

In every iteration, we fix M and solve U by minimization of the cost function L(R, U, M), then we fix U and solve M by the minimization of 
the cost function similarly. The iteration stops until a certain stopping criteria is met.

To solve the matrix U when M is given, each user's feature vector is calculated by resolving a regularized linear least square error function 
using the items the user has rated and their feature vectors:

      1/2 * d(L(R,U,M)) / d(U[u,f]) = 0 

Similary, when M is updated, we resolve a regularized linear least square error function using feature vectors of the users that have rated the 
item and their feature vectors:

      1/2 * d(L(R,U,M)) / d(M[i,f]) = 0

The ALSWRFactorizer class is a non-distributed implementation of ALSWR using multi-threading to dispatch the computation among several threads.
Mahout also offers a [parallel map-reduce implementation](https://mahout.apache.org/users/recommender/intro-als-hadoop.html).

<a name="MatrixFactorization-Reference"></a>
# Reference:

[Stochastic gradient descent](http://en.wikipedia.org/wiki/Stochastic_gradient_descent)
    
[ALSWR](http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08%28submitted%29.pdf)

