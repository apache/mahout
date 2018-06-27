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

package org.apache.mahout.cf.taste.example.bookcrossing;

import org.apache.commons.cli2.OptionException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.example.TasteOptionParser;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public final class BookCrossingBooleanRecommenderEvaluatorRunner {

  private static final Logger log = LoggerFactory.getLogger(BookCrossingBooleanRecommenderEvaluatorRunner.class);

  private BookCrossingBooleanRecommenderEvaluatorRunner() {
    // do nothing
  }

  public static void main(String... args) throws IOException, TasteException, OptionException {
    RecommenderIRStatsEvaluator evaluator = new GenericRecommenderIRStatsEvaluator();
    File ratingsFile = TasteOptionParser.getRatings(args);
    DataModel model =
        ratingsFile == null ? new BookCrossingDataModel(true) : new BookCrossingDataModel(ratingsFile, true);

    IRStatistics evaluation = evaluator.evaluate(
        new BookCrossingBooleanRecommenderBuilder(),
        new BookCrossingDataModelBuilder(),
        model,
        null,
        3,
        Double.NEGATIVE_INFINITY,
        1.0);

    log.info(String.valueOf(evaluation));
  }

}
