/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math;

import java.util.Iterator;

public interface VectorIterable extends Iterable<MatrixSlice> {

  Iterator<MatrixSlice> iterateAll();

  int numSlices();

  int numRows();

  int numCols();

  /**
   * Return a new vector with cardinality equal to getNumRows() of this matrix which is the matrix product of the
   * recipient and the argument
   *
   * @param v a vector with cardinality equal to getNumCols() of the recipient
   * @return a new vector (typically a DenseVector)
   * @throws CardinalityException if this.getNumRows() != v.size()
   */
  Vector times(Vector v);

  /**
   * Convenience method for producing this.transpose().times(this.times(v)), which can be implemented with only one pass
   * over the matrix, without making the transpose() call (which can be expensive if the matrix is sparse)
   *
   * @param v a vector with cardinality equal to getNumCols() of the recipient
   * @return a new vector (typically a DenseVector) with cardinality equal to that of the argument.
   * @throws CardinalityException if this.getNumCols() != v.size()
   */
  Vector timesSquared(Vector v);

}
