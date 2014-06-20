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

import org.apache.mahout.math.Vector;
import org.apache.mahout.h2obindings.H2OHelper;
import org.apache.mahout.h2obindings.drm.H2OBCast;

import water.*;
import water.fvec.*;

public class Ax {
  /* Calculate Ax (where x is an in-core Vector) */
  public static Frame Ax(Frame A, Vector x) {
    final H2OBCast<Vector> bx = new H2OBCast<Vector>(x);
    class MRTaskAx extends MRTask<MRTaskAx> {
      public void map(Chunk chks[], NewChunk nc) {
        Vector x = bx.value();
        for (int r = 0; r < chks[0]._len; r++) {
          double v = 0;
          for (int c = 0; c < chks.length; c++) {
            v += (chks[c].at0(r) * x.getQuick(c));
          }
          nc.addNum(v);
        }
      }
    }
    return new MRTaskAx().doAll(1, A).outputFrame(A.names(), A.domains());
  }
}
