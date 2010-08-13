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

package org.apache.mahout.math.hadoop.similarity.vector;

import java.util.Iterator;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.similarity.Cooccurrence;

/**
 * abstract base implementation of {@link DistributedVectorSimilarity}
 */
public abstract class AbstractDistributedVectorSimilarity implements DistributedVectorSimilarity {

  /**
   * ensures that the computed similarity is in [-1,1]
   */
  @Override
  public final double similarity(int rowA, int rowB, Iterable<Cooccurrence> cooccurrences, double weightOfVectorA,
      double weightOfVectorB, int numberOfColumns) {

    double result = doComputeResult(rowA, rowB, cooccurrences, weightOfVectorA, weightOfVectorB, numberOfColumns);

    if (result < -1.0) {
      result = -1.0;
    } else if (result > 1.0) {
      result = 1.0;
    }
    return result;
  }

  /**
   * computes the number of elements in the {@link Iterable}
   */
  protected static int countElements(Iterable<?> iterable) {
    return countElements(iterable.iterator());
  }

  /**
   * computes the number of elements in the {@link Iterator}
   */
  protected static int countElements(Iterator<?> iterator) {
    int count = 0;
    while (iterator.hasNext()) {
      count++;
      iterator.next();
    }
    return count;
  }

  /**
   * do the actual similarity computation
   *
   * @see DistributedVectorSimilarity#similarity(int, int, Iterable, double, double, int)
   */
  protected abstract double doComputeResult(int rowA,
                                            int rowB,
                                            Iterable<Cooccurrence> cooccurrences,
                                            double weightOfVectorA,
                                            double weightOfVectorB,
                                            int numberOfColumns);

  /**
   * vectors have no weight (NaN) by default, subclasses may override this
   */
  @Override
  public double weight(Vector v) {
    return Double.NaN;
  }

}
