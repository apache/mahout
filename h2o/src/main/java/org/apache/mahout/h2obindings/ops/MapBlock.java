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
import org.apache.mahout.h2obindings.H2OBlockMatrix;
import org.apache.mahout.h2obindings.drm.H2ODrm;

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import java.io.Serializable;
import java.util.Arrays;

import scala.reflect.ClassTag;

public class MapBlock {
  public static <K,R> H2ODrm exec(H2ODrm DrmA, int ncol, Object bmf, final boolean is_r_str,
                                  final ClassTag<K> k, final ClassTag<R> r) {
    Frame A = DrmA.frame;
    Vec keys = DrmA.keys;

    class MRTaskBMF extends MRTask<MRTaskBMF> {
      Serializable bmf;
      Vec labels;
      MRTaskBMF(Object _bmf, Vec _labels) {
        /* BlockMapFun does not implement Serializable,
           but Scala closures are _always_ Serializable.

           So receive the object as a plain Object (else
           compilation fails) and typcast it with conviction,
           that Scala always tags the actually generated
           closure functions with Serializable.
         */
        bmf = (Serializable)_bmf;
        labels = _labels;
      }

      private Matrix blockify(Chunk chks[]) {
        return new H2OBlockMatrix(chks);
      }

      private void deblockify(Matrix out, NewChunk ncs[]) {
        // assert (out.colSize() == ncs.length)
        for (int c = 0; c < out.columnSize(); c++) {
          for (int r = 0; r < out.rowSize(); r++) {
            ncs[c].addNum(out.getQuick(r, c));
          }
        }
      }

      /*
        Input:
        chks.length == A.numCols()

        Output:
        ncs.length == (A.numCols() + 1) if String keyed
                      (A.numCols() + 0) if Int or Long keyed

        First A.numCols() ncs[] elements are fed back the output
        of bmf() output's _2 in deblockify()

        If String keyed, then MapBlockHelper.exec() would have
        filled in the Strings into ncs[ncol] already
      */
      public void map(Chunk chks[], NewChunk ncs[]) {
        long start = chks[0].start();
        NewChunk nclabel = is_r_str ? ncs[ncs.length - 1] : null;
        deblockify(MapBlockHelper.exec(bmf, blockify(chks), start, labels, nclabel, k, r), ncs);
        // assert chks[i]._len == ncs[j]._len
      }
    }

    int ncol_res = ncol + (is_r_str ? 1 : 0);
    Frame fmap = new MRTaskBMF(bmf, keys).doAll(ncol_res, A).outputFrame(null, null);
    Vec vmap = null;
    if (is_r_str) {
      /* If output was String keyed, then the last Vec in fmap is the String vec.
         If so, peel it out into a separate Vec (vmap) and set fmap to be the
         Frame with just the first ncol Vecs
      */
      vmap = fmap.vecs()[ncol];
      fmap = new Frame(Arrays.copyOfRange(fmap.vecs(), 0, ncol));
    }
    return new H2ODrm(fmap, vmap);
  }
}
