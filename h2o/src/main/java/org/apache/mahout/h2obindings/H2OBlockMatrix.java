/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.AbstractMatrix;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.flavor.MatrixFlavor;

import water.fvec.Chunk;

/**
 * A Matrix implementation to represent a vertical Block of DRM.
 *
 * Creation of the matrix is an O(1) operation with negligible
 * overhead, and will remain so as long as the matrix is only
 * read from (no modifications).
 *
 * On the first modification, create a copy on write Matrix and
 * all further operations happen on this cow matrix.
 *
 * The benefit is, mapBlock() closures which never modify the
 * input matrix save on the copy overhead.
 */
public class H2OBlockMatrix extends AbstractMatrix {
  /** Backing chunks which store the original matrix data */
  private Chunk chks[];
  /** Copy on write matrix created on demand when original matrix is modified */
  private Matrix cow;

  /** Class constructor. */
  public H2OBlockMatrix(Chunk chks[]) {
    super(chks[0].len(), chks.length);
    this.chks = chks;
  }

  /**
   * Internal method to create the copy on write matrix.
   *
   * Once created, all further operations are performed on the CoW matrix
   */
  private void cow() {
    if (cow != null) {
      return;
    }

    if (chks[0].isSparse()) {
      cow = new SparseMatrix(chks[0].len(), chks.length);
    } else {
      cow = new DenseMatrix(chks[0].len(), chks.length);
    }

    for (int c = 0; c < chks.length; c++) {
      for (int r = 0; r < chks[0].len(); r++) {
        cow.setQuick(r, c, chks[c].atd(r));
      }
    }
  }

  @Override
  public void setQuick(int row, int col, double val) {
    cow();
    cow.setQuick(row, col, val);
  }

  @Override
  public Matrix like(int nrow, int ncol) {
    if (chks[0].isSparse()) {
      return new SparseMatrix(nrow, ncol);
    } else {
      return new DenseMatrix(nrow, ncol);
    }
  }

  @Override
  public Matrix like() {
    if (chks[0].isSparse()) {
      return new SparseMatrix(rowSize(), columnSize());
    } else {
      return new DenseMatrix(rowSize(), columnSize());
    }
  }

  @Override
  public double getQuick(int row, int col) {
    if (cow != null) {
      return cow.getQuick(row, col);
    } else {
      return chks[col].atd(row);
    }
  }

  @Override
  public Matrix assignRow(int row, Vector v) {
    cow();
    cow.assignRow(row, v);
    return cow;
  }

  @Override
  public Matrix assignColumn(int col, Vector v) {
    cow();
    cow.assignColumn(col, v);
    return cow;
  }

  @Override
  public MatrixFlavor getFlavor() {
    if (cow != null) {
      return cow.getFlavor();
    } else if (chks[0].isSparse()) {
      return MatrixFlavor.SPARSELIKE;
    } else {
      return MatrixFlavor.DENSELIKE;
    }
  }
}
