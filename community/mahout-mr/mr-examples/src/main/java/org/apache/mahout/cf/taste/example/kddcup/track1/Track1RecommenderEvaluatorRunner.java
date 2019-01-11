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

package org.apache.mahout.cf.taste.example.kddcup.track1;

import java.io.File;
import java.io.IOException;

import org.apache.commons.cli2.OptionException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.example.TasteOptionParser;
import org.apache.mahout.cf.taste.example.kddcup.KDDCupDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Track1RecommenderEvaluatorRunner {

  private static final Logger log = LoggerFactory.getLogger(Track1RecommenderEvaluatorRunner.class);

  private Track1RecommenderEvaluatorRunner() {
  }
  
  public static void main(String... args) throws IOException, TasteException, OptionException {
    File dataFileDirectory = TasteOptionParser.getRatings(args);
    if (dataFileDirectory == null) {
      throw new IllegalArgumentException("No data directory");
    }
    if (!dataFileDirectory.exists() || !dataFileDirectory.isDirectory()) {
      throw new IllegalArgumentException("Bad data file directory: " + dataFileDirectory);
    }
    Track1RecommenderEvaluator evaluator = new Track1RecommenderEvaluator(dataFileDirectory);
    DataModel model = new KDDCupDataModel(KDDCupDataModel.getTrainingFile(dataFileDirectory));
    double evaluation = evaluator.evaluate(new Track1RecommenderBuilder(),
      null,
      model,
      Float.NaN,
      Float.NaN);
    log.info(String.valueOf(evaluation));
  }
  
}
