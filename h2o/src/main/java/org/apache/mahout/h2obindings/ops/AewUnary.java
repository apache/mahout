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
import scala.Function1;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;

import java.io.Serializable;

/**
 * MapBlock operator.
 */
public class AewUnary {
  /**
   * Execute a UnaryFunc on each element of a DRM. Create a new DRM
   * with the new values.
   *
   * @param drmA DRM representing matrix A.
   * @param f UnaryFunc f, that accepts and Double and returns a Double.
   * @param evalZeros Whether or not to execute function on zeroes (in case of sparse DRM).
   * @return new DRM constructed from mapped values of drmA through f.
   */
  public static H2ODrm exec(H2ODrm drmA, Object f, final boolean evalZeros) {

    Frame A = drmA.frame;
    Vec keys = drmA.keys;
    final int ncol = A.numCols();

    /**
     * MRTask to execute fn on all elements.
     */
    class MRTaskAewUnary extends MRTask<MRTaskAewUnary> {
      Serializable fn;
      MRTaskAewUnary(Object _fn) {
        fn = (Serializable)_fn;
      }
      public void map(Chunk chks[], NewChunk ncs[]) {
        for (int c = 0; c < chks.length; c++) {
          Chunk chk = chks[c];
          Function1 f = (Function1) fn;
          int ChunkLen = chk.len();

          if (!evalZeros && chk.isSparse()) {
            /* sparse and skip zeros */
            int prev_offset = -1;
            for (int r = chk.nextNZ(-1); r < ChunkLen; r = chk.nextNZ(prev_offset)) {
              if (r - prev_offset > 1)
                ncs[c].addZeros(r - prev_offset - 1);
              ncs[c].addNum((double)f.apply(chk.atd(r)));
              prev_offset = r;
            }
            if (ChunkLen - prev_offset > 1)
              ncs[c].addZeros(chk._len - prev_offset - 1);
          } else {
            /* dense or non-skip zeros */
            for (int r = 0; r < ChunkLen; r++) {
              ncs[c].addNum((double)f.apply(chk.atd(r)));
            }
          }
        }
      }
    }

    Frame fmap = new MRTaskAewUnary(f).doAll(ncol, A).outputFrame(null, null);

    return new H2ODrm(fmap, keys);
  }
}
