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

package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

public final class GenericRecommenderIRStatsEvaluatorImplTest extends TasteTestCase {

  public void testEvaluate() throws Exception {
    DataModel model = getDataModel();
    RecommenderBuilder builder = new RecommenderBuilder() {
      @Override
      public Recommender buildRecommender(DataModel dataModel) throws TasteException {
        return new SlopeOneRecommender(dataModel);
      }
    };
    RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();
    IRStatistics stats = evaluator.evaluate(builder, null, model, null, 1, 0.2, 1.0);
    assertNotNull(stats);
    assertEquals(0.75, stats.getPrecision(), EPSILON);
    assertEquals(0.75, stats.getRecall(), EPSILON);
    assertEquals(0.75, stats.getF1Measure(), EPSILON);
  }

}