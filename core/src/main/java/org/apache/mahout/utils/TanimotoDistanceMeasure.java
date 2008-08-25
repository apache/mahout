package org.apache.mahout.utils;

/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.Matrix;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.WeightedDistanceMeasure;
import org.apache.hadoop.mapred.JobConf;

import java.util.HashSet;
import java.util.Set;


/**
 * Tanimoto coefficient implementation.
 *
 * http://en.wikipedia.org/wiki/Jaccard_index
 *
 */
public class TanimotoDistanceMeasure extends WeightedDistanceMeasure {

  public TanimotoDistanceMeasure() {
  }

  /**
   * Calculates the distance between two vectors.
   *
   * ((a^2 + b^2 - ab) / ab) - 1;
   *
   * @param vector0
   * @param vector1
   * @return 0 for perfect match, > 0 for greater distance
   */
  public double distance(Vector vector0, Vector vector1) {

    // this whole distance measurent thing
    // should be evaluated using an intermediate vector and BinaryFunction or something?
    
    Set<Integer> featuresSeen = new HashSet<Integer>((int)((vector0.size() + vector1.size()) * 0.75));

    double ab = 0;
    double a2 = 0;
    double b2 = 0;

    Double a;
    Double b;
    Double weight;

    for (Vector.Element feature : vector0) {
      if (!featuresSeen.add(feature.index())) {

        a = feature.get();

        b = vector1.get(feature.index());
        if (b == null) {
          b = 0d;
        }

        weight = getWeights() == null ? 1d : getWeights().get(feature.index());

        ab += a * b * weight;
        a2 += a * a * weight;
        b2 += b * b * weight;
      }
    }


    for (Vector.Element feature : vector1) {
      if (!featuresSeen.add(feature.index())) {

        a = vector0.get(feature.index());
        if (a == null) {
          a = 0d;
        }

        b = feature.get();

        weight = getWeights() == null ? 1d : getWeights().get(feature.index());

        ab += a * b * weight;
        a2 += a * a * weight;
        b2 += b * b * weight;
      }
    }

    return (float)((a2 + b2 - ab) / ab) - 1;
  }

}

