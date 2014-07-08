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
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.h2obindings.H2OHelper;
import org.apache.mahout.h2obindings.drm.H2OBCast;

import water.*;
import water.fvec.*;
import scala.Tuple2;

public class Atx {
  /* Calculate A'x (where x is an in-core Vector) */
  public static Tuple2<Frame,Vec> Atx(Tuple2<Frame,Vec> TA, Vector x) {
    Frame A = TA._1();
    final H2OBCast<Vector> bx = new H2OBCast<Vector>(x);
    class MRTaskAtx extends MRTask<MRTaskAtx> {
      double _atx[];
      public void map(Chunk chks[]) {
        Vector x = bx.value();
        long start = chks[0].start();
        _atx = new double[chks.length];
        for (int r = 0; r < chks[0].len(); r++) {
          double d = x.getQuick((int)start + r);
          for (int c = 0; c < chks.length; c++) {
            _atx[c] += (chks[c].at0(r) * d);
          }
        }
      }
      public void reduce(MRTaskAtx other) {
        for (int i = 0; i < _atx.length; i++)
          _atx[i] += other._atx[i];
      }
    }
    Vector v = new DenseVector(new MRTaskAtx().doAll(A)._atx);
    Matrix m = new DenseMatrix(A.numCols(), 1);
    m.assignColumn(0, v);
    return H2OHelper.frame_from_matrix(m, -1, -1);
  }
}
