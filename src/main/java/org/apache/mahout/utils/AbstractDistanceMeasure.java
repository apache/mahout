package org.apache.mahout.utils;

/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.CardinalityException;

/**
 * Subclasses <b>must</b> implement
 * either {@link #distance(Float[], Float[])}
 * or {@link #distance(org.apache.mahout.matrix.Vector, org.apache.mahout.matrix.Vector)}
 */
public abstract class AbstractDistanceMeasure implements DistanceMeasure {


  public float distance(Float[] p1, Float[] p2) {
    double[] d1 = new double[p1.length];
    for (int i = 0; i < p1.length; i++) {
      d1[i] = p1[i];
    }
    double[] d2 = new double[p2.length];
    for (int i = 0; i < p2.length; i++) {
      d2[i] = p2[i];
    }
    return (float) distance(new DenseVector(d1), new DenseVector(d2));
  }


  public double distance(Vector v1, Vector v2) throws CardinalityException {
    Float[] f1 = new Float[v1.cardinality()];
    for (Vector.Element e : v1) {
      f1[e.index()] = (float)e.get();
    }
    Float[] f2 = new Float[v2.cardinality()];
    for (Vector.Element e : v2) {
      f2[e.index()] = (float)e.get();
    }
    return distance(f1, f2);
  }
}
