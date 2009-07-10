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
import org.apache.mahout.cf.taste.model.User;

/** <p>Tests {@link LogLikelihoodSimilarity}.</p> */
public final class LogLikelihoodSimilarityTest extends SimilarityTestCase {

  public void testCorrelation() throws Exception {
    User user1 = getUser("test1", 1.0, 1.0);
    User user2 = getUser("test2", 1.0, null, 1.0);
    User user3 = getUser("test3", null, null, 1.0, 1.0, 1.0);
    User user4 = getUser("test4", 1.0, 1.0, 1.0, 1.0, 1.0);
    User user5 = getUser("test5", null, 1.0, 1.0, 1.0, 1.0);
    DataModel dataModel = getDataModel(user1, user2, user3, user4, user5);

    double correlation = new LogLikelihoodSimilarity(dataModel).
        itemSimilarity(dataModel.getItem("1"), dataModel.getItem("0"));
    assertCorrelationEquals(0.12160727029227925, correlation);

    correlation = new LogLikelihoodSimilarity(dataModel).
        itemSimilarity(dataModel.getItem("0"), dataModel.getItem("1"));
    assertCorrelationEquals(0.12160727029227925, correlation);

    correlation = new LogLikelihoodSimilarity(dataModel).
        itemSimilarity(dataModel.getItem("2"), dataModel.getItem("1"));
    assertCorrelationEquals(0.5423213660693733, correlation);

    correlation = new LogLikelihoodSimilarity(dataModel).
        itemSimilarity(dataModel.getItem("2"), dataModel.getItem("3"));
    assertCorrelationEquals(0.6905400104897509, correlation);

    correlation = new LogLikelihoodSimilarity(dataModel).
        itemSimilarity(dataModel.getItem("3"), dataModel.getItem("4"));
    assertCorrelationEquals(0.8706358464330881, correlation);

    correlation = new LogLikelihoodSimilarity(dataModel).
        itemSimilarity(dataModel.getItem("4"), dataModel.getItem("3"));
    assertCorrelationEquals(0.8706358464330881, correlation);
  }

  public void testRefresh() {
    // Make sure this doesn't throw an exception
    new LogLikelihoodSimilarity(getDataModel()).refresh(null);
  }

}