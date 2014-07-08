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

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.h2obindings.H2OHelper;
import org.apache.mahout.h2obindings.drm.H2OBCast;

import water.*;
import water.fvec.*;
import scala.Tuple2;

public class TimesRightMatrix {

  private static Frame AinCoreB_diagonal(final Frame A, Vector d) {
    final H2OBCast<Vector> bd = new H2OBCast<Vector>(d);

    class MRTaskAinCoreB extends MRTask<MRTaskAinCoreB> {
      public void map(Chunk chks[], NewChunk ncs[]) {
        Vector D = bd.value();
        for (int c = 0; c < ncs.length; c++) {
          for (int r = 0; r < chks[0].len(); r++) {
            double v = (chks[c].at0(r) * D.getQuick(c));
            ncs[c].addNum(v);
          }
        }
      }
    }
    return new MRTaskAinCoreB().doAll(d.size(), A).outputFrame(null,null);
  }

  private static Frame AinCoreB_common(final Frame A, Matrix b) {
    final H2OBCast<Matrix> bb = new H2OBCast<Matrix>(b);

    class MRTaskAinCoreB extends MRTask<MRTaskAinCoreB> {
      public void map(Chunk chks[], NewChunk ncs[]) {
        Matrix B = bb.value();
        for (int c = 0; c < ncs.length; c++) {
          for (int r = 0; r < chks[0].len(); r++) {
            double v = 0;
            for (int i = 0; i < chks.length; i++) {
              v += (chks[i].at0(r) * B.getQuick(i, c));
            }
            ncs[c].addNum(v);
          }
        }
      }
    }
    return new MRTaskAinCoreB().doAll(b.columnSize(), A).outputFrame(null,null);
  }

  /* Multiple with in-core Matrix */
  public static Tuple2<Frame,Vec> TimesRightMatrix(Tuple2<Frame,Vec> TA, Matrix B) {
    Frame A = TA._1();
    Vec VA = TA._2();
    Frame AinCoreB;
    if (B instanceof DiagonalMatrix)
      AinCoreB = AinCoreB_diagonal(A, B.viewDiagonal());
    else
      AinCoreB = AinCoreB_common(A, B);

    return new Tuple2<Frame,Vec>(AinCoreB, VA);
  }
}
