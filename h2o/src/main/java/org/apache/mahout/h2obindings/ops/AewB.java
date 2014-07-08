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

public class AewB {
  /* Element-wise DRM-DRM operations */
  public static Tuple2<Frame,Vec> AewB(Tuple2<Frame,Vec> AT, Tuple2<Frame,Vec> BT, final String op) {
    final Frame A = AT._1();
    final Frame B = BT._1();
    Vec VA = AT._2();

    class MRTaskAewB extends MRTask<MRTaskAewB> {
      private double opfn (String op, double a, double b) {
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
        long start = chks[0].start();
        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[0].len(); r++) {
            ncs[c].addNum(opfn(op, chks[c].at0(r), B.vecs()[c].at(start+r)));
          }
        }
      }
    }
    Frame AewB = new MRTaskAewB().doAll(A.numCols(), A).outputFrame(A.names(), A.domains());

    return new Tuple2<Frame,Vec>(AewB, VA);
  }
}
