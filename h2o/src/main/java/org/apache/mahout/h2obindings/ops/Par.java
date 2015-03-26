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

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.fvec.NewChunk;
import water.parser.ValueString;

import org.apache.mahout.h2obindings.H2OHelper;
import org.apache.mahout.h2obindings.drm.H2ODrm;

/**
 * Parallelize operator.
 */
public class Par {
  /**
   * (re)Parallelize DRM data according to new partitioning hints.
   *
   * @param drmA Input DRM containing data.
   * @param min Hint of minimum number of partitions to parallelize, if not -1.
   * @param exact Hint of exact number of partitions to parallelize, if not -1.
   * @return new DRM holding the same data but parallelized according to new hints.
   */
  public static H2ODrm exec(H2ODrm drmA, int min, int exact) {
    final Frame frin = drmA.frame;
    final Vec vin = drmA.keys;

    // First create a new empty Frame with the required partitioning
    Frame frout = H2OHelper.emptyFrame(frin.numRows(), frin.numCols(), min, exact);
    Vec vout = null;

    if (vin != null) {
      // If String keyed, then run an MRTask on the new frame, and also
      // creat yet another 1-column newer frame for the re-orged String keys.
      // The new String Vec will therefore be similarly partitioned as the
      // new Frame.
      //
      // vout is finally collected by calling anyVec() on outputFrame(),
      // as it is the only column in the output frame.
      vout = new MRTask() {
          public void map(Chunk chks[], NewChunk nc) {
            int chunkSize = chks[0].len();
            Vec vins[] = frin.vecs();
            long start = chks[0].start();
            ValueString vstr = new ValueString();

            for (int r = 0; r < chunkSize; r++) {
              for (int c = 0; c < chks.length; c++) {
                chks[c].set(r, vins[c].at(start + r));
              }
              nc.addStr(vin.atStr(vstr, start + r));
            }
          }
        }.doAll(1, frout).outputFrame(null, null).anyVec();
    } else {
      // If not String keyed, then run and MRTask on the new frame, and
      // just pull in right elements from frin
      new MRTask() {
        public void map(Chunk chks[]) {
          int chunkSize = chks[0].len();
          Vec vins[] = frin.vecs();
          long start = chks[0].start();

          for (int r = 0; r < chunkSize; r++) {
            for (int c = 0; c < chks.length; c++) {
              chks[c].set(r, vins[c].at(start + r));
            }
          }
        }
      }.doAll(frout);
    }

    return new H2ODrm(frout, vout);
  }
}
