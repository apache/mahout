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

import org.junit.Test;

/**
 * tests {@link DistributedUncenteredCosineVectorSimilarity}
 */
public final class DistributedUncenteredCosineVectorSimilarityTest
    extends DistributedVectorSimilarityTestCase {

  @Test
  public void testUncenteredCosineSimilarity() throws Exception {

    assertSimilar(new DistributedUncenteredCosineVectorSimilarity(),
        asVector(0, 0, 0, 0, 1),
        asVector(0, 1, 1, 1, 1), 5, 1.0);

    assertSimilar(new DistributedUncenteredCosineVectorSimilarity(),
        asVector(0, 1),
        asVector(1, 0), 2, Double.NaN);

    assertSimilar(new DistributedUncenteredCosineVectorSimilarity(),
        asVector(1, 0),
        asVector(1, 0), 2, 1.0);

    assertSimilar(new DistributedUncenteredCosineVectorSimilarity(),
        asVector(1, 1, 2),
        asVector(3, 5, 0), 3, 0.9701425);
  }

}
