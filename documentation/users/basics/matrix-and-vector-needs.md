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
title: Matrix and Vector Needs

    
---

<a name="MatrixandVectorNeeds-Intro"></a>
# Intro

Most ML algorithms require the ability to represent multidimensional data
concisely and to be able to easily perform common operations on that data.
MAHOUT-6 introduced Vector and Matrix datatypes of arbitrary cardinality,
along with a set of common operations on their instances. Vectors and
matrices are provided with sparse and dense implementations that are memory
resident and are suitable for manipulating intermediate results within
mapper, combiner and reducer implementations. They are not intended for
applications requiring vectors or matrices that exceed the size of a single
JVM, though such applications might be able to utilize them within a larger
organizing framework.

<a name="MatrixandVectorNeeds-Background"></a>
## Background

See [http://mail-archives.apache.org/mod_mbox/lucene-mahout-dev/200802.mbox/browser](http://mail-archives.apache.org/mod_mbox/lucene-mahout-dev/200802.mbox/browser)

<a name="MatrixandVectorNeeds-Vectors"></a>
## Vectors

Mahout supports a Vector interface that defines the following operations over all implementation classes: assign, cardinality, copy, divide, dot, get, haveSharedCells, like, minus, normalize, plus, set, size, times, toArray, viewPart, zSum and cross. The class DenseVector implements vectors as a double[](.html)
 that is storage and access efficient. The class SparseVector implements
vectors as a HashMap<Integer, Double> that is surprisingly fast and
efficient. For sparse vectors, the size() method returns the current number
of elements whereas the cardinality() method returns the number of
dimensions it holds. An additional VectorView class allows views of an
underlying vector to be specified by the viewPart() method. See the
JavaDocs for more complete definitions.

<a name="MatrixandVectorNeeds-Matrices"></a>
## Matrices

Mahout also supports a Matrix interface that defines a similar set of operations over all implementation classes: assign, assignColumn, assignRow, cardinality, copy, divide, get, haveSharedCells, like, minus, plus, set, size, times, transpose, toArray, viewPart and zSum. The class DenseMatrix implements matrices as a double[](.html)
[] that is storage and access efficient. The class SparseRowMatrix
implements matrices as a Vector[] holding the rows of the matrix in a
SparseVector, and the symmetric class SparseColumnMatrix implements
matrices as a Vector[] holding the columns in a SparseVector. Each of these
classes can quickly produce a given row or column, respectively. A fourth
class SparseMatrix, uses a HashMap<Integer, Vector> which is also a
SparseVector. For sparse matrices, the size() method returns an int\[2\]
containing the actual row and column sizes whereas the cardinality() method
returns an int\[2\] with the number of dimensions of each. An additional
MatrixView class allows views of an underlying matrix to be specified by
the viewPart() method. See the JavaDocs for more complete definitions.

The Matrix interface does not currently provide invert or determinant
methods, though these are desirable. It is arguable that the
implementations of SparseRowMatrix and SparseColumnMatrix ought to use the
HashMap<Integer, Vector> implementations and that SparseMatrix should
instead use a HashMap<Integer, HashMap<Integer, Double>>. Other forms of
sparse matrices can also be envisioned that support different storage and
access characteristics. Because the arguments of assignColumn and assignRow
operations accept all forms of Vector, it is possible to construct
instances of sparse matrices containing dense rows or columns. See the
JavaDocs for more complete definitions.

For applications like PageRank/TextRank, iterative approaches to calculate
eigenvectors would also be useful. Batching of row/column operations would
also be useful, such as perhaps assignRow or assighColumn accepting
UnaryFunction and BinaryFunction arguments.


<a name="MatrixandVectorNeeds-Ideas"></a>
## Ideas

As Vector and Matrix implementations are currently memory-resident, very
large instances greater than available memory are not supported. An
extended set of implementations that use HBase (BigTable) in Hadoop to
represent their instances would facilitate applications requiring such
large collections.  
See [MAHOUT-6](https://issues.apache.org/jira/browse/MAHOUT-6)
See [Hama](http://wiki.apache.org/hadoop/Hama)


<a name="MatrixandVectorNeeds-References"></a>
## References

Have a look at the old parallel computing libraries like [ScalaPACK](http://www.netlib.org/scalapack/)
, others
