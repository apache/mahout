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

package org.apache.mahout.h2obindings.ops;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.h2obindings.H2OHelper;
import org.apache.mahout.h2obindings.drm.H2OBCast;
import org.apache.mahout.h2obindings.drm.H2ODrm;

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Chunk;
import water.util.ArrayUtils;

/**
 * Calculate A'x (where x is an in-core Vector)
 */
public class Atx {
  /**
   * Perform A'x operation with a DRM and an in-core Vector to create a new DRM.
   *
   * @param drmA DRM representing matrix A.
   * @param x in-core Mahout Vector.
   * @return new DRM containing A'x.
   */
  public static H2ODrm exec(H2ODrm drmA, Vector x) {
    Frame A = drmA.frame;
    final H2OBCast<Vector> bx = new H2OBCast<>(x);

    // A'x is computed into atx[] with an MRTask on A (with
    // x available as a Broadcast
    //
    // x.size() == A.numRows()
    // atx.length == chks.length == A.numCols()
    class MRTaskAtx extends MRTask<MRTaskAtx> {
      double atx[];
      public void map(Chunk chks[]) {
        int chunkSize = chks[0].len();
        Vector x = bx.value();
        long start = chks[0].start();

        atx = new double[chks.length];
        for (int r = 0; r < chunkSize; r++) {
          double d = x.getQuick((int)start + r);
          for (int c = 0; c < chks.length; c++) {
            atx[c] += (chks[c].atd(r) * d);
          }
        }
      }
      public void reduce(MRTaskAtx other) {
        ArrayUtils.add(atx, other.atx);
      }
    }

    // Take the result in .atx[], and convert into a Frame
    // using existing helper functions (creating a Matrix
    // along the way for the Helper)
    Vector v = new DenseVector(new MRTaskAtx().doAll(A).atx);
    Matrix m = new DenseMatrix(A.numCols(), 1);
    m.assignColumn(0, v);
    return H2OHelper.drmFromMatrix(m, -1, -1);
  }
}
