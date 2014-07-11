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

public class AewScalar {
  /* Element-wise DRM-DRM operations */
  public static Tuple2<Frame,Vec> AewScalar(final Tuple2<Frame,Vec> TA, final double s, final String op) {
    Frame A = TA._1();
    Vec VA = TA._2();
    int AewScalar_cols = A.numCols();

    /* AewScalar is written into ncs[] with an MRTask on A, and therefore will
       be similarly partitioned as A.
    */
    Frame AewScalar = new MRTask() {
        private double opfn(String op, double a, double b) {
          if (a == 0.0 && b == 0.0)
            return 0.0;
          if (op.equals("+"))
            return a + b;
          else if (op.equals("-"))
            return a - b;
          else if (op.equals("*"))
            return a * b;
          else if (op.equals("/"))
            return a / b;
          return 0.0;
        }
        public void map(Chunk chks[], NewChunk ncs[]) {
          int chunk_size = chks[0].len();
          long start = chks[0].start();

          for (int c = 0; c < chks.length; c++) {
            for (int r = 0; r < chunk_size; r++) {
              ncs[c].addNum(opfn(op, chks[c].at0(r), s));
            }
          }
        }
      }.doAll(AewScalar_cols, A).outputFrame(null, null);

    /* Carry forward labels of A blindly into ABt */
    return new Tuple2<Frame,Vec>(AewScalar, VA);
  }
}
