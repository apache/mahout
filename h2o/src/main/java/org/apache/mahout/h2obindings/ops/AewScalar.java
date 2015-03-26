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

import org.apache.mahout.h2obindings.drm.H2ODrm;

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.fvec.NewChunk;

/**
 * Element-wise DRM-Scalar operations
 */
public class AewScalar {
  /**
   * Perform element-wise operation on a DRM with a Scalar to create a new DRM.
   *
   * @param drmA DRM representing matrix A.
   * @param s Scalar value represented as a double.
   * @param op Element-wise operator encoded as a String.
   * @return new DRM containing A (element-wise) B.
   */
  public static H2ODrm exec(H2ODrm drmA, final double s, final String op) {
    Frame A = drmA.frame;
    Vec keys = drmA.keys;
    int AewScalar_cols = A.numCols();

    // AewScalar is written into ncs[] with an MRTask on A, and therefore will
    // be similarly partitioned as A.
    Frame AewScalar = new MRTask() {
        private double opfn(String op, double a, double b) {
          if (a == 0.0 && b == 0.0) {
            return 0.0;
          }
          if (op.equals("+")) {
            return a + b;
          } else if (op.equals("-")) {
            return a - b;
          } else if (op.equals("*")) {
            return a * b;
          } else if (op.equals("/")) {
            return a / b;
          }
          return 0.0;
        }
        public void map(Chunk chks[], NewChunk ncs[]) {
          int chunkSize = chks[0].len();
          long start = chks[0].start();

          for (int c = 0; c < chks.length; c++) {
            for (int r = 0; r < chunkSize; r++) {
              ncs[c].addNum(opfn(op, chks[c].atd(r), s));
            }
          }
        }
      }.doAll(AewScalar_cols, A).outputFrame(null, null);

    // Carry forward labels of A blindly into ABt
    return new H2ODrm(AewScalar, keys);
  }
}
