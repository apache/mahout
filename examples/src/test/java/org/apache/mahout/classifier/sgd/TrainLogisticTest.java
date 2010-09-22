/*
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

package org.apache.mahout.classifier.sgd;

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.io.Files;
import com.google.common.io.Resources;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.examples.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Set;

public class TrainLogisticTest extends MahoutTestCase {
  Splitter onWhiteSpace = Splitter.on(CharMatcher.BREAKING_WHITESPACE).trimResults().omitEmptyStrings();
  @Test
  public void testMain() throws IOException {
    String outputFile = "./model";
    String inputFile = "donut.csv";
    String[] args = Iterables.toArray(onWhiteSpace.split(
      "--input " +
        inputFile +
        " --output " +
        outputFile +
        " --target color --categories 2 " +
        "--predictors x y --types numeric --features 20 --passes 100 --rate 50 "), String.class);
    TrainLogistic.main(args);
    LogisticModelParameters lmp = TrainLogistic.getParameters();
    assertEquals(1e-4, lmp.getLambda(), 1e-9);
    assertEquals(20, lmp.getNumFeatures());
    assertEquals(true, lmp.useBias());
    assertEquals("color", lmp.getTargetVariable());
    CsvRecordFactory csv = lmp.getCsvRecordFactory();
    assertEquals("[1, 2]", Sets.newTreeSet(csv.getTargetCategories()).toString());
    assertEquals("[Intercept Term, x, y]", Sets.newTreeSet(csv.getPredictors()).toString());


    AbstractVectorClassifier model = TrainLogistic.getModel();
    ModelDissector md = new ModelDissector(2);
    List<String> data = Resources.readLines(Resources.getResource(inputFile), Charsets.UTF_8);
    for (String line : data.subList(1, data.size())) {
      Vector v = new DenseVector(lmp.getNumFeatures());
      csv.getTraceDictionary().clear();
      csv.processLine(line, v);
      md.update(v, csv.getTraceDictionary(), model);
    }

    List<ModelDissector.Weight> weights = md.summary(10);
    Set<String> expected = Sets.newHashSet("x", "y", "Intercept Term");
    for (ModelDissector.Weight weight : weights) {
      assertTrue(expected.remove(weight.getFeature()));
    }
    assertEquals(0, expected.size());
    System.out.printf("%s\n", weights);
  }
}
