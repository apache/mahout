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

public class TimesRightMatrix {

  private static Frame AinCoreB_diagonal(final Frame A, Vector d) {
    final H2OBCast<Vector> bd = new H2OBCast<Vector>(d);
    /* XXX: create AinCore like A */
    Frame AinCoreB = H2OHelper.empty_frame (A.numRows(), d.size(), 0);


    class MRTaskAinCoreB extends MRTask<MRTaskAinCoreB> {
      public void map(Chunk chks[]) {
        Vector D = bd.value();
        long start = chks[0]._start;
        for (int c = 0; c < ncs.length; c++) {
          for (int r = 0; r < chks[0]._len; r++) {
            double v = (A.vecs()[c].at(start+r) * D.getQuick(c));
            chks[c].set0(r, v);
          }
        }
      }
    }
    new MRTaskAinCoreB().doAll(AinCoreB);
    return AinCoreB;
  }

  private static Frame AinCoreB_common(final Frame A, Matrix b) {
    final H2OBCast<Matrix> bb = new H2OBCast<Matrix>(b);
    /* XXX: create AinCore like A */
    Frame AinCoreB = H2OHelper.empty_frame (A.numRows(), b.columnSize(), 0);

    class MRTaskAinCoreB extends MRTask<MRTaskAinCoreB> {
      public void map(Chunk chks[]) {
        Matrix B = bb.value();
        long start = chks[0]._start;
        for (int c = 0; c < ncs.length; c++) {
          for (int r = 0; r < chks[0]._len; r++) {
            double v = 0;
            for (int i = 0; i < chks.length; i++) {
              v += (A.vecs()[i].at(start+r) * B.getQuick(i, c));
            }
            chks[c].set0(r, v);
          }
        }
      }
    }
    new MRTaskAinCoreB().doAll(AinCoreB);
    return AinCoreB;
  }

  /* Multiple with in-core Matrix */
  public static Frame TimesRightMatrix(Frame A, Matrix B) {
    Frame AinCoreB;
    if (B instanceof DiagonalMatrix)
      AinCoreB = AinCoreB_diagonal(A, B.viewDiagonal());
    else
      AinCoreB = AinCoreB_common(A, B);

    return AinCoreB;
  }
}
