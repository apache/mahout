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

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.fvec.NewChunk;

import scala.Tuple2;

public class AtB {
  /* Calculate A'B */
  public static Tuple2<Frame,Vec> AtB(Tuple2<Frame,Vec> TA, Tuple2<Frame,Vec> TB) {
    final Frame A = TA._1();
    final Frame B = TB._1();

    /* First create an empty frame of the required dimensions */
    Frame AtB = H2OHelper.empty_frame(A.numCols(), B.numCols(), -1, -1);

    /* Execute MRTask on the new Frame, and fill each cell (initially 0) by
       computing appropriate values from A and B.

       chks.length == B.numCols()
    */
    new MRTask() {
      public void map(Chunk chks[]) {
        int chunk_size = chks[0].len();
        long start = chks[0].start();
        long A_rows = A.numRows();
        Vec A_vecs[] = A.vecs();
        Vec B_vecs[] = B.vecs();

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chunk_size; r++) {
            double v = 0;
            for (long i = 0; i < A_rows; i++) {
              v += (A_vecs[(int)(start+r)].at(i) * B_vecs[c].at(i));
            }
            chks[c].set0(r, v);
          }
        }
      }
    }.doAll(AtB);

    /* AtB is NOT similarly partitioned as A, drop labels */
    return new Tuple2<Frame,Vec>(AtB, null);
  }
}
