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

import water.*;
import water.fvec.*;
import scala.Tuple2;
import org.apache.mahout.h2obindings.H2OHelper;

public class Par {
  public static Tuple2<Frame,Vec> exec(Tuple2<Frame,Vec> TA, int min, int exact) {
    final Frame frin = TA._1();
    final Vec vin = TA._2();
    Frame frout = H2OHelper.empty_frame (frin.numRows(), frin.numCols(), min, exact);
    Vec vout = null;

    class MRParVecTask extends MRTask<MRParVecTask> {
      public void map(Chunk chks[], NewChunk nc) {
        Vec vins[] = frin.vecs();
        for (int r = 0; r < chks[0].len(); r++) {
          for (int c = 0; c < chks.length; c++) {
            chks[c].set0(r, vins[c].at(chks[0].start() + r));
          }
          nc.addStr(vin.atStr(chks[0].start() + r));
        }
      }
    }

    class MRParTask extends MRTask<MRParTask> {
      public void map(Chunk chks[]) {
        Vec vins[] = frin.vecs();
        for (int r = 0; r < chks[0].len(); r++) {
          for (int c = 0; c < chks.length; c++) {
            chks[c].set0(r, vins[c].at(chks[0].start() + r));
          }
        }
      }
    }

    if (vout != null) {
      vout = new MRParVecTask().doAll(1, frout).outputFrame(null, null).anyVec();
    } else {
      new MRParTask().doAll(frout);
    }
    return new Tuple2<Frame,Vec> (frout, vout);
  }
}
