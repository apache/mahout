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
import org.apache.mahout.h2obindings.H2OHelper;
import org.apache.mahout.h2obindings.drm.H2OBCast;
import org.apache.mahout.h2obindings.drm.H2ODrm;

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.fvec.NewChunk;

public class Ax {
  /* Calculate Ax (where x is an in-core Vector) */
  public static H2ODrm Ax(H2ODrm DrmA, Vector x) {
    Frame A = DrmA.frame;
    Vec keys = DrmA.keys;
    final H2OBCast<Vector> bx = new H2OBCast<Vector>(x);

    /* Ax is written into nc (single element, not array) with an MRTask on A,
       and therefore will be similarly partitioned as A.

       x.size() == A.numCols() == chks.length
    */
    Frame Ax = new MRTask() {
        public void map(Chunk chks[], NewChunk nc) {
          int chunk_size = chks[0].len();
          Vector x = bx.value();

          for (int r = 0; r < chunk_size; r++) {
            double v = 0;
            for (int c = 0; c < chks.length; c++) {
              v += (chks[c].at0(r) * x.getQuick(c));
            }
            nc.addNum(v);
          }
        }
      }.doAll(1, A).outputFrame(null, null);

    /* Carry forward labels of A blindly into ABt */
    return new H2ODrm(Ax, keys);
  }
}
