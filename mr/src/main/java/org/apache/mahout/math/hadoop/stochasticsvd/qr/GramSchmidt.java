/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.math.hadoop.stochasticsvd.qr;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;

/**
 * Gram Schmidt quick helper.
 */
public final class GramSchmidt {

  private GramSchmidt() {
  }

  public static void orthonormalizeColumns(Matrix mx) {

    int n = mx.numCols();

    for (int c = 0; c < n; c++) {
      Vector col = mx.viewColumn(c);
      for (int c1 = 0; c1 < c; c1++) {
        Vector viewC1 = mx.viewColumn(c1);
        col.assign(col.minus(viewC1.times(viewC1.dot(col))));

      }
      final double norm2 = col.norm(2);
      col.assign(new DoubleFunction() {
        @Override
        public double apply(double x) {
          return x / norm2;
        }
      });
    }
  }

}
