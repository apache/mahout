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

package org.apache.mahout.cf.taste.example.grouplens;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * <p>A simple example "runner" class which will evaluate the performance of the current
 * implementation of {@link GroupLensRecommender}.</p>
 */
public final class GroupLensRecommenderEvaluatorRunner {

  private static final Logger log = LoggerFactory.getLogger(GroupLensRecommenderEvaluatorRunner.class);

  private GroupLensRecommenderEvaluatorRunner() {
    // do nothing
  }

  public static void main(String... args) throws IOException, TasteException {
    RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
    double evaluation = evaluator.evaluate(new GroupLensRecommenderBuilder(),
                                           null,
                                           new GroupLensDataModel(),
                                           0.9,
                                           0.1);
    log.info(String.valueOf(evaluation));
  }

}
