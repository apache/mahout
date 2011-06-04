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

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import com.google.common.io.Resources;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.examples.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TrainLogisticTest extends MahoutTestCase {

  @Test
  public void example13_1() throws Exception {
    String outputFile = getTestTempFile("model").getAbsolutePath();

    String trainOut = runMain(TrainLogistic.class, new String[]{
      "--input", "donut.csv",
      "--output", outputFile,
      "--target", "color", "--categories", "2",
      "--predictors", "x", "y",
      "--types", "numeric",
      "--features", "20",
      "--passes", "100",
      "--rate", "50"
    });
    assertTrue(trainOut.contains("x -0.7"));
    assertTrue(trainOut.contains("y -0.4"));

    LogisticModelParameters lmp = TrainLogistic.getParameters();
    assertEquals(1.0e-4, lmp.getLambda(), 1.0e-9);
    assertEquals(20, lmp.getNumFeatures());
    assertTrue(lmp.useBias());
    assertEquals("color", lmp.getTargetVariable());
    CsvRecordFactory csv = lmp.getCsvRecordFactory();
    assertEquals("[1, 2]", Sets.newTreeSet(csv.getTargetCategories()).toString());
    assertEquals("[Intercept Term, x, y]", Sets.newTreeSet(csv.getPredictors()).toString());

    // verify model by building dissector
    AbstractVectorClassifier model = TrainLogistic.getModel();
    List<String> data = Resources.readLines(Resources.getResource("donut.csv"), Charsets.UTF_8);
    Map<String, Double> expectedValues = ImmutableMap.of("x", -0.7, "y", -0.43, "Intercept Term", -0.15);
    verifyModel(lmp, csv, data, model, expectedValues);

    // test saved model
    InputStream in = new FileInputStream(new File(outputFile));
    try {
      LogisticModelParameters lmpOut = LogisticModelParameters.loadFrom(in);
      CsvRecordFactory csvOut = lmpOut.getCsvRecordFactory();
      csvOut.firstLine(data.get(0));
      OnlineLogisticRegression lrOut = lmpOut.createRegression();
      verifyModel(lmpOut, csvOut, data, lrOut, expectedValues);
    } finally {
      Closeables.closeQuietly(in);
    }

    String output = runMain(RunLogistic.class, new String[]{
        "--input", "donut.csv",
        "--model", outputFile,
        "--auc",
        "--confusion"
    });
    assertTrue(output.contains("AUC = 0.57"));
    assertTrue(output.contains("confusion: [[27.0, 13.0], [0.0, 0.0]]"));
  }

  @Test
  public void example13_2() throws Exception {
    String outputFile = getTestTempFile("model").getAbsolutePath();
    String trainOut = runMain(TrainLogistic.class, new String[]{
        "--input", "donut.csv",
        "--output", outputFile,
        "--target", "color",
        "--categories", "2",
        "--predictors", "x", "y", "a", "b", "c",
        "--types", "numeric",
        "--features", "20",
        "--passes", "100",
        "--rate", "50"
    });

    assertTrue(trainOut.contains("a 0."));
    assertTrue(trainOut.contains("b -1."));
    assertTrue(trainOut.contains("c -25."));

    String output = runMain(RunLogistic.class, new String[]{
        "--input", "donut.csv",
        "--model", outputFile,
        "--auc",
        "--confusion"
    });
    assertTrue(output.contains("AUC = 1.00"));

    String heldout = runMain(RunLogistic.class, new String[]{
        "--input", "donut-test.csv",
        "--model", outputFile,
        "--auc",
        "--confusion"
    });
    assertTrue(heldout.contains("AUC = 0.9"));
  }

  /**
   * Runs a class with a public static void main method.  We assume that there is an accessible
   * field named "output" that we can change to redirect output.
   *
   *
   * @param clazz   contains the main method.
   * @param args    contains the command line arguments
   * @return The contents to standard out as a string.
   * @throws IOException                   Not possible, but must be declared.
   * @throws NoSuchFieldException          If there isn't an output field.
   * @throws IllegalAccessException        If the output field isn't accessible by us.
   * @throws NoSuchMethodException         If there isn't a main method.
   * @throws InvocationTargetException     If the main method throws an exception.
   */
  private static String runMain(Class<?> clazz, String[] args)
    throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
    ByteArrayOutputStream trainOutput = new ByteArrayOutputStream();
    PrintStream printStream = new PrintStream(trainOutput);

    try {
      Field outputField = clazz.getDeclaredField("output");
      Method main = clazz.getMethod("main", args.getClass());

      outputField.set(null, printStream);
      Object[] argList = {args};
      main.invoke(null, argList);
      return new String(trainOutput.toByteArray(), Charsets.UTF_8);
    } finally {
      Closeables.closeQuietly(printStream);
    }
  }

  private static void verifyModel(LogisticModelParameters lmp,
                                  RecordFactory csv,
                                  List<String> data,
                                  AbstractVectorClassifier model,
                                  Map<String, Double> expectedValues) {
    ModelDissector md = new ModelDissector();
    for (String line : data.subList(1, data.size())) {
      Vector v = new DenseVector(lmp.getNumFeatures());
      csv.getTraceDictionary().clear();
      csv.processLine(line, v);
      md.update(v, csv.getTraceDictionary(), model);
    }

    // check right variables are present
    List<ModelDissector.Weight> weights = md.summary(10);
    Set<String> expected = Sets.newHashSet(expectedValues.keySet());
    for (ModelDissector.Weight weight : weights) {
      assertTrue(expected.remove(weight.getFeature()));
      assertEquals(expectedValues.get(weight.getFeature()), weight.getWeight(), 0.1);
    }
    assertEquals(0, expected.size());
  }
}
