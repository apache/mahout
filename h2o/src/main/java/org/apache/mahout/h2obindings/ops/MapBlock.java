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

import water.*;
import water.fvec.*;
import java.io.Serializable;

import scala.reflect.ClassTag;

public class MapBlock {
  public static <K,R> Frame exec(Frame A, int ncol, Object bmf, final ClassTag<K> k, final ClassTag<R> r) {
    class MRTaskBMF extends MRTask<MRTaskBMF> {
      Serializable _bmf;
      MRTaskBMF(Object bmf) {
        /* BlockMapFun does not implement Serializable,
           but Scala closures are _always_ Serializable.

           So receive the object as a plain Object (else
           compilation fails) and typcast it with conviction,
           that Scala always tags the actually generated
           closure functions with Serializable.
         */
        _bmf = (Serializable)bmf;
      }

      private Matrix blockify (Chunk chks[]) {
        return new H2OBlockMatrix(chks);
      }

      private void deblockify (Matrix out, NewChunk ncs[]) {
        // assert (out.colSize() == ncs.length)
        for (int c = 0; c < out.columnSize(); c++) {
          for (int r = 0; r < out.rowSize(); r++) {
            ncs[c].addNum(out.getQuick(r, c));
          }
        }
      }

      public void map(Chunk chks[], NewChunk ncs[]) {
        deblockify(MapBlockHelper.exec(_bmf, blockify(chks), chks[0]._start, k, r), ncs);
        // assert chks[i]._len == ncs[j]._len
      }
    }
    return new MRTaskBMF(bmf).doAll(ncol, A).outputFrame(A.names(), A.domains());
  }
}
