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

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.h2obindings.drm.H2OBCast;
import org.apache.mahout.h2obindings.drm.H2ODrm;

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.fvec.NewChunk;

/**
 * Multiple DRM with in-core Matrix
 */
public class TimesRightMatrix {
  /**
   * Multiply a DRM with an in-core Matrix to create a new DRM.
   *
   * @param drmA DRM representing matrix A.
   * @param B in-core Mahout Matrix.
   * @return new DRM containing drmA times B.
   */
  public static H2ODrm exec(H2ODrm drmA, Matrix B) {
    Frame A = drmA.frame;
    Vec keys = drmA.keys;
    Frame AinCoreB = null;

    if (B instanceof DiagonalMatrix) {
      AinCoreB = execDiagonal(A, B.viewDiagonal());
    } else {
      AinCoreB = execCommon(A, B);
    }

    return new H2ODrm(AinCoreB, keys);
  }

  /**
   * Multiply Frame A with in-core diagonal Matrix (whose diagonal Vector is d)
   *
   * A.numCols() == d.size()
   */
  private static Frame execDiagonal(final Frame A, Vector d) {
    final H2OBCast<Vector> bd = new H2OBCast<Vector>(d);

    return new MRTask() {
      public void map(Chunk chks[], NewChunk ncs[]) {
        Vector D = bd.value();
        int chunkSize = chks[0].len();

        for (int c = 0; c < ncs.length; c++) {
          for (int r = 0; r < chunkSize; r++) {
            double v = (chks[c].atd(r) * D.getQuick(c));
            ncs[c].addNum(v);
          }
        }
      }
    }.doAll(d.size(), A).outputFrame(null, null);
  }

  /**
   * Multiply Frame A with in-core Matrix b
   *
   * A.numCols() == b.rowSize()
   */
  private static Frame execCommon(final Frame A, Matrix b) {
    final H2OBCast<Matrix> bb = new H2OBCast<Matrix>(b);

    return new MRTask() {
      public void map(Chunk chks[], NewChunk ncs[]) {
        Matrix B = bb.value();
        int chunkSize = chks[0].len();

        for (int c = 0; c < ncs.length; c++) {
          for (int r = 0; r < chunkSize; r++) {
            double v = 0;
            for (int i = 0; i < chks.length; i++) {
              v += (chks[i].atd(r) * B.getQuick(i, c));
            }
            ncs[c].addNum(v);
          }
        }
      }
    }.doAll(b.columnSize(), A).outputFrame(null, null);
  }
}
