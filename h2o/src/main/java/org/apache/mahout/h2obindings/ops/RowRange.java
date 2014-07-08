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

import scala.collection.immutable.Range;

import water.*;
import water.fvec.*;
import scala.Tuple2;

public class RowRange {
  /* Filter operation */
  public static Tuple2<Frame,Vec> RowRange(Tuple2<Frame,Vec> TA, Range r) {
    Frame A = TA._1();
    Vec VA = TA._2();

    class MRTaskFilter extends MRTask<MRTaskFilter> {
      Range _r;
      MRTaskFilter(Range r) {
        _r = r;
      }
      public void map(Chunk chks[], NewChunk ncs[]) {
        if (chks[0].start() > _r.end() || (chks[0].start() + chks[0].len()) < _r.start())
          return;

        for (int r = 0; r < chks[0].len(); r++) {
          if (!_r.contains (chks[0].start() + r))
            continue;

          for (int c = 0; c < chks.length; c++)
            ncs[c].addNum(chks[c].at0(r));
        }
      }
    }
    Frame Arr = new MRTaskFilter(r).doAll(A.numCols(), A).outputFrame(A.names(), A.domains());
    Vec Vrr = null;
    if (VA != null) {
      class MRTaskStrFilter extends MRTask<MRTaskStrFilter> {
        Range _r;
        MRTaskStrFilter(Range r) {
          _r = r;
        }
        public void map(Chunk chk, NewChunk nc) {
          if (chk.start() > _r.end() || (chk.start() + chk.len()) < _r.start())
            return;

          for (int r = 0; r < chk.len(); r++) {
            if (!_r.contains (chk.start() + r))
              continue;

            nc.addStr(chk.atStr0(r));
          }
        }
      }
      Vrr = new MRTaskStrFilter(r).doAll(1, VA).outputFrame(null,null).vecs()[0];
    }

    return new Tuple2<Frame,Vec>(Arr,Vrr);
  }
}
