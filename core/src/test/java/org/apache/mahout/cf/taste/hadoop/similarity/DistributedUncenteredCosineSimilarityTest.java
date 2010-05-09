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

package org.apache.mahout.cf.taste.hadoop.similarity;

/**
 * test for {@link DistributedUncenteredCosineSimilarity}
 */
public class DistributedUncenteredCosineSimilarityTest extends
    DistributedItemSimilarityTestCase {

  public void testUncenteredCosine() throws Exception {

    assertSimilar(new DistributedUncenteredCosineSimilarity(), 2,
        new Float[] { Float.NaN, Float.NaN, Float.NaN, Float.NaN, 1.0f },
        new Float[] { Float.NaN, 1.0f, 1.0f, 1.0f, 1.0f }, 1.0);

    assertSimilar(new DistributedUncenteredCosineSimilarity(), 2,
        new Float[] { Float.NaN, 1.0f },
        new Float[] { 1.0f, Float.NaN }, Double.NaN);

    assertSimilar(new DistributedUncenteredCosineSimilarity(), 2,
        new Float[] { 1.0f, Float.NaN },
        new Float[] { 1.0f, Float.NaN }, 1.0);

    assertSimilar(new DistributedUncenteredCosineSimilarity(), 2,
        new Float[] { 1.0f, 1.0f, 2.0f },
        new Float[] { 3.0f, 5.0f, Float.NaN }, 0.970142);
  }

}
