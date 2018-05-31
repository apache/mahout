/*
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

import java.io.Serializable;
import java.util.Comparator;

/**
 * Orders {@link WeightedVector} by {@link WeightedVector#getWeight()}.
 */
public final class WeightedVectorComparator implements Comparator<WeightedVector>, Serializable {

  private static final double DOUBLE_EQUALITY_ERROR = 1.0e-8;

  @Override
  public int compare(WeightedVector a, WeightedVector b) {
    if (a == b) {
      return 0;
    }
    double aWeight = a.getWeight();
    double bWeight = b.getWeight();
    int r = Double.compare(aWeight, bWeight);
    if (r != 0 && Math.abs(aWeight - bWeight) >= DOUBLE_EQUALITY_ERROR) {
      return r;
    }
    double diff = a.minus(b).norm(1);
    if (diff < 1.0e-12) {
      return 0;
    }
    for (Vector.Element element : a.all()) {
      r = Double.compare(element.get(), b.get(element.index()));
      if (r != 0) {
        return r;
      }
    }
    return 0;
  }

}
