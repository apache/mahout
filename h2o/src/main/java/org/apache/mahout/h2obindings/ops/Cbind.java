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

public class Cbind {
  private static Tuple2<Frame, Vec> zip(final Frame fra, final Vec va, final Frame frb, final Vec vb) {
    Vec vecs[] = new Vec[fra.vecs().length + frb.vecs().length];
    int d = 0;
    for (Vec vfra : fra.vecs())
      vecs[d++] = vfra;
    for (Vec vfrb : frb.vecs())
      vecs[d++] = vfrb;
    Frame fr = new Frame(vecs);
    return new Tuple2<Frame,Vec> (fr, va);
  }

  private static Tuple2<Frame, Vec> join(final Frame fra, final Vec va, final Frame frb, final Vec vb) {
    Vec bvecs[] = new Vec[frb.vecs().length];

    for (int i = 0; i < bvecs.length; i++)
      bvecs[i] = fra.anyVec().makeZero();

    new MRTask() {
      public void map(Chunk chks[]) {
        long start = chks[0].start();
        for (int r = 0; r < chks[0].len(); r++) {
          for (int c = 0; c < chks.length; c++) {
            // assert va.atStr(start+r) == vb.atStr(start+r)
            chks[c].set0(r, frb.vecs()[c].at(start + r));
          }
        }
      }
    }.doAll(bvecs);

    Vec vecs[] = new Vec[fra.vecs().length + frb.vecs().length];
    int d = 0;
    for (Vec vfra : fra.vecs())
      vecs[d++] = vfra;
    for (Vec vfrb : bvecs)
      vecs[d++] = vfrb;
    Frame fr = new Frame(vecs);
    return new Tuple2<Frame,Vec> (fr, va);
  }

  public static Tuple2<Frame,Vec> Cbind(Tuple2<Frame,Vec> TA, Tuple2<Frame,Vec> TB) {
    Frame fra = TA._1();
    Vec va = TA._2();
    Frame frb = TB._1();
    Vec vb = TB._2();

    if (fra.anyVec().group() == frb.anyVec().group())
      return zip(fra, va, frb, vb);
    else
      return join(fra, va, frb, vb);
  }
}
