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

package org.apache.mahout.cf.taste.example.netflix;

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

public final class NetflixRecommenderEvaluatorRunner {
  
  private static final Logger log = LoggerFactory.getLogger(NetflixRecommenderEvaluatorRunner.class);
  
  private NetflixRecommenderEvaluatorRunner() {
    // do nothing
  }
  
  public static void main(String... args) throws IOException, TasteException, OptionException {
    RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
    File ratingsFile = TasteOptionParser.getRatings(args);
    if (ratingsFile != null) {
      DataModel model = new NetflixDataModel(ratingsFile, true);
      double evaluation = evaluator.evaluate(new NetflixRecommenderBuilder(), null, model, 0.9, 0.1);
      log.info(String.valueOf(evaluation));
    } else {
      log.error("Netflix Recommender needs a ratings file to work. Please provide it with the -i command line option.");
    }
  }
  
}
