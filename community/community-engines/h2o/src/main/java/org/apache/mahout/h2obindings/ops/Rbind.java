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
 * R-like rbind like operator, on two DRMs
 */
public class Rbind {
  /**
   * Combine the rows of two DRMs A and B to create a new DRM.
   *
   * @param drmA DRM representing matrix A.
   * @param drmB DRM representing matrix B.
   * @return new DRM containing rows of B below A.
   */
  public static H2ODrm exec(H2ODrm drmA, H2ODrm drmB) {
    final Frame fra = drmA.frame;
    final Vec keysa = drmA.keys;
    final Frame frb = drmB.frame;
    final Vec keysb = drmB.keys;

    // Create new frame and copy A's data at the top, and B's data below.
    // Create the frame in the same VectorGroup as A, so A's data does not
    // cross the wire during copy. B's data could potentially cross the wire.
    Frame frbind = H2OHelper.emptyFrame(fra.numRows() + frb.numRows(), fra.numCols(),
            -1, -1, fra.anyVec().group());
    Vec keys = null;

    MRTask task = new MRTask() {
        public void map(Chunk chks[], NewChunk nc) {
          Vec A_vecs[] = fra.vecs();
          Vec B_vecs[] = frb.vecs();
          long A_rows = fra.numRows();
          long B_rows = frb.numRows();
          long start = chks[0].start();
          int chunkSize = chks[0].len();
          ValueString vstr = new ValueString();

          for (int r = 0; r < chunkSize; r++) {
            for (int c = 0; c < chks.length; c++) {
              if (r + start < A_rows) {
                chks[c].set(r, A_vecs[c].at(r + start));
                if (keysa != null) {
                  nc.addStr(keysa.atStr(vstr, r + start));
                }
              } else {
                chks[c].set(r, B_vecs[c].at(r + start - A_rows));
                if (keysb != null) {
                  nc.addStr(keysb.atStr(vstr, r + start - A_rows));
                }
              }
            }
          }
        }
      };

    if (keysa == null) {
      keys = task.doAll(1, frbind).outputFrame(null, null).anyVec();
    } else {
      task.doAll(frbind);
    }

    return new H2ODrm(frbind, keys);
  }
}
