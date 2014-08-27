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

import scala.collection.immutable.Range;

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.fvec.NewChunk;
import water.parser.ValueString;

import org.apache.mahout.h2obindings.drm.H2ODrm;

public class RowRange {
  /* Filter operation */
  public static H2ODrm RowRange(H2ODrm drmA, final Range R) {
    Frame A = drmA.frame;
    Vec keys = drmA.keys;

    /* Run a filtering MRTask on A. If row number falls within R.start() and
       R.end(), then the row makes it into the output
    */
    Frame Arr = new MRTask() {
        public void map(Chunk chks[], NewChunk ncs[]) {
          int chunk_size = chks[0].len();
          long chunk_start = chks[0].start();

          /* First check if the entire chunk even overlaps with R */
          if (chunk_start > R.end() || (chunk_start + chunk_size) < R.start()) {
            return;
          }

          /* This chunk overlaps, filter out just the overlapping rows */
          for (int r = 0; r < chunk_size; r++) {
            if (!R.contains(chunk_start + r)) {
              continue;
            }

            for (int c = 0; c < chks.length; c++) {
              ncs[c].addNum(chks[c].at0(r));
            }
          }
        }
      }.doAll(A.numCols(), A).outputFrame(null, null);

    Vec Vrr = (keys == null) ? null : new MRTask() {
        /* This is a String keyed DRM. Do the same thing as above,
           but this time just one column of Strings.
        */
        public void map(Chunk chk, NewChunk nc) {
          int chunk_size = chk.len();
          long chunk_start = chk.start();
          ValueString vstr = new ValueString();

          if (chunk_start > R.end() || (chunk_start + chunk_size) < R.start()) {
            return;
          }

          for (int r = 0; r < chunk_size; r++) {
            if (!R.contains(chunk_start + r)) {
              continue;
            }

            nc.addStr(chk.atStr0(vstr, r));
          }
        }
      }.doAll(1, keys).outputFrame(null, null).anyVec();

    return new H2ODrm(Arr, Vrr);
  }
}
