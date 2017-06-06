/*
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

package org.apache.mahout.math.flavor;

/** A set of matrix structure properties that I denote as "flavor" (by analogy to quarks) */
public interface MatrixFlavor {

  /**
   * Whether matrix is backed by a native system -- such as java memory, lapack/atlas, Magma etc.
   */
  BackEnum getBacking();

  /**
   * Structure flavors
   */
  TraversingStructureEnum getStructure() ;

  boolean isDense();

  /**
   * This default for {@link org.apache.mahout.math.DenseMatrix}-like structures
   */
  MatrixFlavor DENSELIKE = new FlavorImpl(BackEnum.JVMMEM, TraversingStructureEnum.ROWWISE, true);
  /**
   * This is default flavor for {@link org.apache.mahout.math.SparseRowMatrix}-like.
   */
  MatrixFlavor SPARSELIKE = new FlavorImpl(BackEnum.JVMMEM, TraversingStructureEnum.ROWWISE, false);

  /**
   * This is default flavor for {@link org.apache.mahout.math.SparseMatrix}-like structures, i.e. sparse matrix blocks,
   * where few, perhaps most, rows may be missing entirely.
   */
  MatrixFlavor SPARSEROWLIKE = new FlavorImpl(BackEnum.JVMMEM, TraversingStructureEnum.SPARSEROWWISE, false);

  /**
   * This is default flavor for {@link org.apache.mahout.math.DiagonalMatrix} and the likes.
   */
  MatrixFlavor DIAGONALLIKE = new FlavorImpl(BackEnum.JVMMEM, TraversingStructureEnum.VECTORBACKED, false);

  final class FlavorImpl implements MatrixFlavor {
    private BackEnum pBacking;
    private TraversingStructureEnum pStructure;
    private boolean pDense;

    public FlavorImpl(BackEnum backing, TraversingStructureEnum structure, boolean dense) {
      pBacking = backing;
      pStructure = structure;
      pDense = dense;
    }

    @Override
    public BackEnum getBacking() {
      return pBacking;
    }

    @Override
    public TraversingStructureEnum getStructure() {
      return pStructure;
    }

    @Override
    public boolean isDense() {
      return pDense;
    }
  }

}
