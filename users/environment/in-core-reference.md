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
layout: default
title: 

   
---

## Mahout-Samsara's In-Core Linear Algebra DSL Reference

#### Imports

The following imports are used to enable Mahout-Samsara's Scala DSL bindings for in-core Linear Algebra:

    import org.apache.mahout.math._
    import scalabindings._
    import RLikeOps._
    
#### Inline initalization

Dense vectors:

    val densVec1: Vector = (1.0, 1.1, 1.2)
    val denseVec2 = dvec(1, 0, 1,1 ,1,2)

Sparse vectors:

    val sparseVec1: Vector = (5 -> 1.0) :: (10 -> 2.0) :: Nil
    val sparseVec1 = svec((5 -> 1.0) :: (10 -> 2.0) :: Nil)

    // to create a vector with specific cardinality
    val sparseVec1 = svec((5 -> 1.0) :: (10 -> 2.0) :: Nil, cardinality = 20)
    
Inline matrix initialization, either sparse or dense, is always done row wise. 

Dense matrices:

    val A = dense((1, 2, 3), (3, 4, 5))
    
Sparse matrices:

    val A = sparse(
              (1, 3) :: Nil,
              (0, 2) :: (1, 2.5) :: Nil
                  )

Diagonal matrix with constant diagonal elements:

    diag(3.5, 10)

Diagonal matrix with main diagonal backed by a vector:

    diagv((1, 2, 3, 4, 5))
    
Identity matrix:

    eye(10)
    
####Slicing and Assigning

Getting a vector element:

    val d = vec(5)

Setting a vector element:
    
    vec(5) = 3.0
    
Getting a matrix element:

    val d = m(3,5)
    
Setting a matrix element:

    M(3,5) = 3.0
    
Getting a matrix row or column:

    val rowVec = M(3, ::)
    val colVec = M(::, 3)
    
Setting a matrix row or column via vector assignment:

    M(3, ::) := (1, 2, 3)
    M(::, 3) := (1, 2, 3)
    
Setting a subslices of a matrix row or column:

    a(0, 0 to 1) = (3, 5)
   
Setting a subslices of a matrix row or column via vector assignment:

    a(0, 0 to 1) := (3, 5)
   
Getting a matrix as from matrix contiguous block:

    val B = A(2 to 3, 3 to 4)
   
Assigning a contiguous block to a matrix:

    A(0 to 1, 1 to 2) = dense((3, 2), (3 ,3))
   
Assigning a contiguous block to a matrix using the matrix assignment operator:

    A(o to 1, 1 to 2) := dense((3, 2), (3, 3))
   
Assignment operator used for copying between vectors or matrices:

    vec1 := vec2
    M1 := M2
   
Assignment operator using assignment through a functional literal for a matrix:

    M := ((row, col, x) => if (row == col) 1 else 0
    
Assignment operator using assignment through a functional literal for a vector:

    vec := ((index, x) => sqrt(x)
    
#### BLAS-like operations

Plus/minus either vector or numeric with assignment or not:

    a + b
    a - b
    a + 5.0
    a - 5.0
    
Hadamard (elementwise) product, either vector or matrix or numeric operands:

    a * b
    a * 0.5

Operations with assignment:

    a += b
    a -= b
    a += 5.0
    a -= 5.0
    a *= b
    a *= 5
   
*Some nuanced rules*: 

1/x in R (where x is a vector or a matrix) is elementwise inverse.  In scala it would be expressed as:

    val xInv = 1 /: x

and R's 5.0 - x would be:
   
    val x1 = 5.0 -: x
    
*note: All assignment operations, including :=, return the assignee just like in C++*:

    a -= b 
    
assigns **a - b** to **b** (in-place) and returns **b**.  Similarly for **a /=: b** or **1 /=: v** 
    

Dot product:

    a dot b
    
Matrix and vector equivalency (or non-equivalency).  **Dangerous, exact equivalence is rarely useful, better to use norm comparisons with an allowance of small errors.**
    
    a === b
    a !== b
    
Matrix multiply:    

    a %*% b
    
Optimized Right Multiply with a diagonal matrix: 

    diag(5, 5) :%*% b
   
Optimized Left Multiply with a diagonal matrix:

    A %*%: diag(5, 5)

Second norm, of a vector or matrix:

    a.norm
    
Transpose:

    val Mt = M.t
    
*note: Transposition is currently handled via view, i.e. updating a transposed matrix will be updating the original.*  Also computing something like `\(\mathbf{X^\top}\mathbf{X}\)`:

    val XtX = X.t %*% X
    
will not therefore incur any additional data copying.

#### Decompositions

Matrix decompositions require an additional import:

    import org.apache.mahout.math.decompositions._


All arguments in the following are matricies.

**Cholesky decomposition**

    val ch = chol(M)
    
**SVD**

    val (U, V, s) = svd(M)
    
**EigenDecomposition**

    val (V, d) = eigen(M)
    
**QR decomposition**

    val (Q, R) = qr(M)
    
**Rank**: Check for rank deficiency (runs rank-revealing QR)

    M.isFullRank
   
**In-core SSVD**

    Val (U, V, s) = ssvd(A, k = 50, p = 15, q = 1)
    
**Solving linear equation systems and matrix inversion:** fully similar to R semantics; there are three forms of invocation:


Solve `\(\mathbf{AX}=\mathbf{B}\)`:

    solve(A, B)
   
Solve `\(\mathbf{Ax}=\mathbf{b}\)`:
  
    solve(A, b)
   
Compute `\(\mathbf{A^{-1}}\)`:

    solve(A)
   
#### Misc

Vector cardinality:

    a.length
    
Matrix cardinality:

    m.nrow
    m.ncol
    
Means and sums:

    m.colSums
    m.colMeans
    m.rowSums
    m.rowMeans
    
Copy-By-Value:

    val b = a cloned
    
#### Random Matrices

`\(\mathcal{U}\)`(0,1) random matrix view:

    val incCoreA = Matrices.uniformView(m, n, seed)

    
`\(\mathcal{U}\)`(-1,1) random matrix view:

    val incCoreA = Matrices.symmetricUniformView(m, n, seed)

`\(\mathcal{N}\)`(-1,1) random matrix view:

    val incCoreA = Matrices.gaussianView(m, n, seed)
    
#### Iterators 

Mahout-Math already exposes a number of iterators.  Scala code just needs the following imports to enable implicit conversions to scala iterators.

    import collection._
    import JavaConversions._
    
Iterating over rows in a Matrix:

    for (row <- m) {
      ... do something with row
    }
    
<!--Iterating over non-zero and all elements of a vector:
*Note that Vector.Element also has some implicit syntatic sugar, e.g to add 5.0 to every non-zero element of a matrix, the following code may be used:*

    for (row <- m; el <- row.nonZero) el = 5.0 + el
    ... or 
    for (row <- m; el <- row.nonZero) el := 5.0 + el
    
Similarly **row.all** produces an iterator over all elements in a row (Vector). 
-->

For more information including information on Mahout-Samsara's out-of-core Linear algebra bindings see: [Mahout Scala Bindings and Mahout Spark Bindings for Linear Algebra Subroutines](http://mahout.apache.org/users/sparkbindings/ScalaSparkBindings.pdf)


