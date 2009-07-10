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

package org.apache.mahout.utils;

import org.apache.mahout.matrix.Vector;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;


/**
 * Tanimoto coefficient implementation.
 *
 * http://en.wikipedia.org/wiki/Jaccard_index
 */
public class TanimotoDistanceMeasure extends WeightedDistanceMeasure {


  /**
   * Calculates the distance between two vectors.
   *
   * ((a^2 + b^2 - ab) / ab) - 1;
   *
   * @return 0 for perfect match, > 0 for greater distance
   */
  @Override
  public double distance(Vector vector0, Vector vector1) {

    // this whole distance measurent thing
    // should be evaluated using an intermediate vector and BinaryFunction or something?

    Set<Integer> featuresSeen = new HashSet<Integer>((int) ((vector0.getNumNondefaultElements() + vector1.getNumNondefaultElements()) * 0.75));

    double ab = 0.0;
    double a2 = 0.0;
    double b2 = 0.0;

    Iterator<Vector.Element> iter = vector0.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element feature = iter.next();
      if (!featuresSeen.add(feature.index())) {

        double a = feature.get();

        double b = vector1.get(feature.index());

        Vector weights = getWeights();
        double weight = weights == null ? 1.0 : weights.get(feature.index());

        ab += a * b * weight;
        a2 += a * a * weight;
        b2 += b * b * weight;
      }
    }


    iter = vector1.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element feature = iter.next();
      if (!featuresSeen.add(feature.index())) {

        double a = vector0.get(feature.index());

        double b = feature.get();

        Vector weights = getWeights();
        double weight = weights == null ? 1.0 : weights.get(feature.index());

        ab += a * b * weight;
        a2 += a * a * weight;
        b2 += b * b * weight;
      }
    }

    return ((a2 + b2 - ab) / ab) - 1.0;
  }

  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return distance(centroid, v); // TODO
  }

}

