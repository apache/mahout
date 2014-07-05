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

public class ABt {
  /* Calculate AB' */
  public static Tuple2<Frame,Vec> ABt(Tuple2<Frame,Vec> TA, Tuple2<Frame,Vec> TB) {
    Frame A = TA._1();
    Vec VA = TA._2();
    final Frame B = TB._1();

    class MRTaskABt extends MRTask<MRTaskABt> {
      public void map(Chunk chks[], NewChunk ncs[]) {
        for (int c = 0; c < ncs.length; c++) {
          for (int r = 0; r < chks[0]._len; r++) {
            double v = 0;
            for (int i = 0; i < chks.length; i++) {
              v += (chks[i].at0(r) * B.vecs()[i].at(c));
            }
            ncs[c].addNum(v);
          }
        }
      }
    }
    Frame ABt = new MRTaskABt().doAll((int)B.numRows(),A).outputFrame(null,null);
    return new Tuple2<Frame,Vec>(ABt, VA);
  }
}
