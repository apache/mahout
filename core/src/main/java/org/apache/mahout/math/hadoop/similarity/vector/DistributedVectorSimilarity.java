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

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.similarity.Cooccurrence;

/**
 * a measure for the pairwise similarity of two rows of a matrix that is suitable for computing that similarity
 * in a distributed way
 *
 * works in 2 steps:
 *  - at first weight() is called for each of the row vectors
 *  - later similarity is called with the previously computed weights as parameters
 *
 */
public interface DistributedVectorSimilarity {

  /**
   * compute the weight (e.g. length) of a vector
   */
  double weight(Vector v);

  /**
   * compute the similarity of a pair of row vectors
   *
   * @param rowA              offset of the first row
   * @param rowB              offset of the second row
   * @param cooccurrences     all column entries where both vectors have a nonZero entry
   * @param weightOfVectorA   the result of {@link DistributedVectorSimilarity#weight(Vector)} for the first row vector
   * @param weightOfVectorB   the result of {@link DistributedVectorSimilarity#weight(Vector)} for the first row vector
   */
  double similarity(int rowA,
                    int rowB,
                    Iterable<Cooccurrence> cooccurrences,
                    double weightOfVectorA,
                    double weightOfVectorB,
                    long numberOfColumns);
}
