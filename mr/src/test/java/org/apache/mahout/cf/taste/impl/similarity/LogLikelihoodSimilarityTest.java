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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.model.DataModel;
import org.junit.Test;

/** <p>Tests {@link LogLikelihoodSimilarity}.</p> */
public final class LogLikelihoodSimilarityTest extends SimilarityTestCase {

  @Test
  public void testCorrelation() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3, 4, 5},
            new Double[][] {
                    {1.0, 1.0},
                    {1.0, null, 1.0},
                    {null, null, 1.0, 1.0, 1.0},
                    {1.0, 1.0, 1.0, 1.0, 1.0},
                    {null, 1.0, 1.0, 1.0, 1.0},
            });

    LogLikelihoodSimilarity similarity = new LogLikelihoodSimilarity(dataModel);

    assertCorrelationEquals(0.12160727029227925, similarity.itemSimilarity(1, 0));
    assertCorrelationEquals(0.12160727029227925, similarity.itemSimilarity(0, 1));

    assertCorrelationEquals(0.5423213660693732, similarity.itemSimilarity(1, 2));
    assertCorrelationEquals(0.5423213660693732, similarity.itemSimilarity(2, 1));

    assertCorrelationEquals(0.6905400104897509, similarity.itemSimilarity(2, 3));
    assertCorrelationEquals(0.6905400104897509, similarity.itemSimilarity(3, 2));

    assertCorrelationEquals(0.8706358464330881, similarity.itemSimilarity(3, 4));
    assertCorrelationEquals(0.8706358464330881, similarity.itemSimilarity(4, 3));
  }

  @Test
  public void testNoSimilarity() throws Exception {

    DataModel dataModel = getDataModel(
        new long[] {1, 2, 3, 4},
        new Double[][] {
                {1.0, null, 1.0, 1.0},
                {1.0, null, 1.0, 1.0},
                {null, 1.0, 1.0, 1.0},
                {null, 1.0, 1.0, 1.0},
        });

    LogLikelihoodSimilarity similarity = new LogLikelihoodSimilarity(dataModel);

    assertCorrelationEquals(Double.NaN, similarity.itemSimilarity(1, 0));
    assertCorrelationEquals(Double.NaN, similarity.itemSimilarity(0, 1));

    assertCorrelationEquals(0.0, similarity.itemSimilarity(2, 3));
    assertCorrelationEquals(0.0, similarity.itemSimilarity(3, 2));
  }

  @Test
  public void testRefresh() {
    // Make sure this doesn't throw an exception
    new LogLikelihoodSimilarity(getDataModel()).refresh(null);
  }

}