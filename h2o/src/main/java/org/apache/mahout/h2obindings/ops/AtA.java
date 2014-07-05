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
import scala.Tuple2;

public class AtA {
  /* Calculate A'A */
  public static Tuple2<Frame,Vec> AtA(Tuple2<Frame,Vec> TA) {
    final Frame A = TA._1();
    Frame AtA = H2OHelper.empty_frame (A.numCols(), A.numCols(), 0);
    class MRTaskAtA extends MRTask<MRTaskAtA> {
      public void map(Chunk chks[]) {
        long start = chks[0]._start;
        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[0]._len; r++) {
            double v = 0;
            for (int i = 0; i < A.numRows(); i++) {
              v += (A.vecs()[(int)(start+r)].at(i) * A.vecs()[c].at(i));
            }
            chks[c].set0(r, v);
          }
        }
      }
    }
    new MRTaskAtA().doAll(AtA);
    return new Tuple2<Frame,Vec>(AtA,null);
  }
}
