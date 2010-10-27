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

import java.io.File;
import java.io.IOException;

import org.apache.commons.cli2.OptionException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.example.TasteOptionParser;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>A simple example "runner" class which will evaluate the performance of the current
 * implementation of {@link GroupLensRecommender}.</p>
 */
public final class GroupLensRecommenderEvaluatorRunner {
  
  private static final Logger log = LoggerFactory.getLogger(GroupLensRecommenderEvaluatorRunner.class);
  
  private GroupLensRecommenderEvaluatorRunner() {
    // do nothing
  }
  
  public static void main(String... args) throws IOException, TasteException, OptionException {
    RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
    File ratingsFile = TasteOptionParser.getRatings(args);
    DataModel model = ratingsFile == null ? new GroupLensDataModel() : new GroupLensDataModel(ratingsFile);
    double evaluation = evaluator.evaluate(new GroupLensRecommenderBuilder(),
      null,
      model,
      0.9,
      0.3);
    log.info(String.valueOf(evaluation));
  }
  
}
