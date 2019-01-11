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

/**
 * Filter operation
 */
public class RowRange {
  /**
   * Filter rows from intput DRM, to include only row indiced included in R.
   *
   * @param drmA Input DRM.
   * @param R Range object specifying the start and end row numbers to filter.
   * @return new DRM with just the filtered rows.
   */
  public static H2ODrm exec(H2ODrm drmA, final Range R) {
    Frame A = drmA.frame;
    Vec keys = drmA.keys;

    // Run a filtering MRTask on A. If row number falls within R.start() and
    // R.end(), then the row makes it into the output
    Frame Arr = new MRTask() {
        public void map(Chunk chks[], NewChunk ncs[]) {
          int chunkSize = chks[0].len();
          long chunkStart = chks[0].start();

          // First check if the entire chunk even overlaps with R
          if (chunkStart > R.end() || (chunkStart + chunkSize) < R.start()) {
            return;
          }

          // This chunk overlaps, filter out just the overlapping rows
          for (int r = 0; r < chunkSize; r++) {
            if (!R.contains(chunkStart + r)) {
              continue;
            }

            for (int c = 0; c < chks.length; c++) {
              ncs[c].addNum(chks[c].atd(r));
            }
          }
        }
      }.doAll(A.numCols(), A).outputFrame(null, null);

    Vec Vrr = (keys == null) ? null : new MRTask() {
        // This is a String keyed DRM. Do the same thing as above,
        // but this time just one column of Strings.
        public void map(Chunk chk, NewChunk nc) {
          int chunkSize = chk.len();
          long chunkStart = chk.start();
          ValueString vstr = new ValueString();

          if (chunkStart > R.end() || (chunkStart + chunkSize) < R.start()) {
            return;
          }

          for (int r = 0; r < chunkSize; r++) {
            if (!R.contains(chunkStart + r)) {
              continue;
            }

            nc.addStr(chk.atStr(vstr, r));
          }
        }
      }.doAll(1, keys).outputFrame(null, null).anyVec();

    return new H2ODrm(Arr, Vrr);
  }
}
