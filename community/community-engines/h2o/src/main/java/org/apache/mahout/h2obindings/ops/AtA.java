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

import org.apache.mahout.h2obindings.H2OHelper;
import org.apache.mahout.h2obindings.drm.H2ODrm;

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;

/**
 * Calculate A'A
 */
public class AtA {
  /**
   * Perform A'A operation on a DRM to create a new DRM.
   *
   * @param drmA DRM representing matrix A.
   * @return new DRM containing A'A.
   */
 public static H2ODrm exec(H2ODrm drmA) {
    final Frame A = drmA.frame;
    // First create an empty Frame of the required dimensions
    Frame AtA = H2OHelper.emptyFrame(A.numCols(), A.numCols(), -1, -1);

    // Execute MRTask on the new Frame, and fill each cell (initially 0) by
    // computing appropriate values from A.
    //
    // chks.length == A.numCols()
    new MRTask() {
      public void map(Chunk chks[]) {
        int chunkSize = chks[0].len();
        long start = chks[0].start();
        Vec A_vecs[] = A.vecs();
        long A_rows = A.numRows();

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chunkSize; r++) {
            double v = 0;
            for (long i = 0; i < A_rows; i++) {
              v += (A_vecs[(int)(start + r)].at(i) * A_vecs[c].at(i));
            }
            chks[c].set(r, v);
          }
        }
      }
    }.doAll(AtA);

    // AtA is NOT similarly partitioned as A, drop labels
    return new H2ODrm(AtA);
  }
}
