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

package org.apache.mahout.math;

import com.google.common.collect.Lists;

import java.util.List;

public final class OrthonormalityVerifier {

  private OrthonormalityVerifier() {
  }

  public static VectorIterable pairwiseInnerProducts(Iterable<MatrixSlice> basis) {
    DenseMatrix out = null;
    for (MatrixSlice slice1 : basis) {
      List<Double> dots = Lists.newArrayList();
      for (MatrixSlice slice2 : basis) {
        dots.add(slice1.vector().dot(slice2.vector()));
      }
      if (out == null) {
        out = new DenseMatrix(dots.size(), dots.size());
      }
      for (int i = 0; i < dots.size(); i++) {
        out.set(slice1.index(), i, dots.get(i));
      }
    }
    return out;
  }

}
