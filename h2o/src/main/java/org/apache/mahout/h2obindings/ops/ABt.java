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

import org.apache.mahout.h2obindings.H2OHelper;

import water.*;
import water.fvec.*;

public class ABt {
  /* Calculate AB' */
  public static Frame ABt(final Frame A, final Frame B) {
    /* XXX - make ABt similar to A */
    Frame ABt = H2OHelper.empty_frame (A.numRows(), (int)B.numRows(), 0);

    class MRTaskABt extends MRTask<MRTaskABt> {
      public void map(Chunk chks[]) {
        long start = chks[0]._start;
        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[0]._len; r++) {
            double v = 0;
            for (int i = 0; i < A.vecs().length; i++) {
              v += (A.vecs()[i].at(start+r) * B.vecs()[i].at(c));
            }
            chks[c].set0(r, v);
          }
        }
      }
    }
    new MRTaskABt().doAll(ABt);
    return ABt;
  }
}
