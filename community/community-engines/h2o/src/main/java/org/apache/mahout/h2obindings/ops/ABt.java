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
 * Calculate AB'
 */
public class ABt {
  /**
   * Calculate AB' on two DRMs to create a new DRM holding the result.
   *
   * @param drmA DRM representing matrix A
   * @param drmB DRM representing matrix B
   * @return new DRM containing AB'
   */
  public static H2ODrm exec(H2ODrm drmA, H2ODrm drmB) {
    Frame A = drmA.frame;
    Vec keys = drmA.keys;
    final Frame B = drmB.frame;
    int ABt_cols = (int)B.numRows();

    // ABt is written into ncs[] with an MRTask on A, and therefore will
    // be similarly partitioned as A.
    //
    // chks.length == A.numCols() (== B.numCols())
    // ncs.length == ABt_cols (B.numRows())
    Frame ABt = new MRTask() {
        public void map(Chunk chks[], NewChunk ncs[]) {
          int chunkSize = chks[0].len();
          Vec B_vecs[] = B.vecs();

          for (int c = 0; c < ncs.length; c++) {
            for (int r = 0; r < chunkSize; r++) {
              double v = 0;
              for (int i = 0; i < chks.length; i++) {
                v += (chks[i].atd(r) * B_vecs[i].at(c));
              }
              ncs[c].addNum(v);
            }
          }
        }
      }.doAll(ABt_cols, A).outputFrame(null, null);

    // Carry forward labels of A blindly into ABt
    return new H2ODrm(ABt, keys);
  }
}
