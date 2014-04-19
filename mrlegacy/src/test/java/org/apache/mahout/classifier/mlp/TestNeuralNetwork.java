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
package org.apache.mahout.classifier.mlp;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.csv.CSVUtils;
import org.apache.mahout.classifier.mlp.NeuralNetwork.TrainingMethod;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Files;

/**
 * Test the functionality of {@link NeuralNetwork}.
 */
public class TestNeuralNetwork extends MahoutTestCase {

  @Test
  public void testReadWrite() throws IOException {
    NeuralNetwork ann = new MultilayerPerceptron();
    ann.addLayer(2, false, "Identity");
    ann.addLayer(5, false, "Identity");
    ann.addLayer(1, true, "Identity");
    ann.setCostFunction("Minus_Squared");
    double learningRate = 0.2;
    double momentumWeight = 0.5;
    double regularizationWeight = 0.05;
    ann.setLearningRate(learningRate).setMomentumWeight(momentumWeight).setRegularizationWeight(regularizationWeight);

    // manually set weights
    Matrix[] matrices = new DenseMatrix[2];
    matrices[0] = new DenseMatrix(5, 3);
    matrices[0].assign(0.2);
    matrices[1] = new DenseMatrix(1, 6);
    matrices[1].assign(0.8);
    ann.setWeightMatrices(matrices);

    // write to file
    String modelFilename = "testNeuralNetworkReadWrite";
    File tmpModelFile = this.getTestTempFile(modelFilename);
    ann.setModelPath(tmpModelFile.getAbsolutePath());
    ann.writeModelToFile();

    // read from file
    NeuralNetwork annCopy = new MultilayerPerceptron(tmpModelFile.getAbsolutePath());
    assertEquals(annCopy.getClass().getSimpleName(), annCopy.getModelType());
    assertEquals(tmpModelFile.getAbsolutePath(), annCopy.getModelPath());
    assertEquals(learningRate, annCopy.getLearningRate(), 0.000001);
    assertEquals(momentumWeight, annCopy.getMomentumWeight(), 0.000001);
    assertEquals(regularizationWeight, annCopy.getRegularizationWeight(), 0.000001);
    assertEquals(TrainingMethod.GRADIENT_DESCENT, annCopy.getTrainingMethod());

    // compare weights
    Matrix[] weightsMatrices = annCopy.getWeightMatrices();
    for (int i = 0; i < weightsMatrices.length; ++i) {
      Matrix expectMat = matrices[i];
      Matrix actualMat = weightsMatrices[i];
      for (int j = 0; j < expectMat.rowSize(); ++j) {
        for (int k = 0; k < expectMat.columnSize(); ++k) {
          assertEquals(expectMat.get(j, k), actualMat.get(j, k), 0.000001);
        }
      }
    }
  }

  /**
   * Test the forward functionality.
   */
  @Test
  public void testOutput() {
    // first network
    NeuralNetwork ann = new MultilayerPerceptron();
    ann.addLayer(2, false, "Identity");
    ann.addLayer(5, false, "Identity");
    ann.addLayer(1, true, "Identity");
    ann.setCostFunction("Minus_Squared").setLearningRate(0.1);

    // intentionally initialize all weights to 0.5
    Matrix[] matrices = new Matrix[2];
    matrices[0] = new DenseMatrix(5, 3);
    matrices[0].assign(0.5);
    matrices[1] = new DenseMatrix(1, 6);
    matrices[1].assign(0.5);
    ann.setWeightMatrices(matrices);

    double[] arr = new double[]{0, 1};
    Vector training = new DenseVector(arr);
    Vector result = ann.getOutput(training);
    assertEquals(1, result.size());

    // second network
    NeuralNetwork ann2 = new MultilayerPerceptron();
    ann2.addLayer(2, false, "Sigmoid");
    ann2.addLayer(3, false, "Sigmoid");
    ann2.addLayer(1, true, "Sigmoid");
    ann2.setCostFunction("Minus_Squared");
    ann2.setLearningRate(0.3);

    // intentionally initialize all weights to 0.5
    Matrix[] matrices2 = new Matrix[2];
    matrices2[0] = new DenseMatrix(3, 3);
    matrices2[0].assign(0.5);
    matrices2[1] = new DenseMatrix(1, 4);
    matrices2[1].assign(0.5);
    ann2.setWeightMatrices(matrices2);

    double[] test = {0, 0};
    double[] result2 = {0.807476};

    Vector vec = ann2.getOutput(new DenseVector(test));
    double[] arrVec = new double[vec.size()];
    for (int i = 0; i < arrVec.length; ++i) {
      arrVec[i] = vec.getQuick(i);
    }
    assertArrayEquals(result2, arrVec, 0.000001);

    NeuralNetwork ann3 = new MultilayerPerceptron();
    ann3.addLayer(2, false, "Sigmoid");
    ann3.addLayer(3, false, "Sigmoid");
    ann3.addLayer(1, true, "Sigmoid");
    ann3.setCostFunction("Minus_Squared").setLearningRate(0.3);

    // intentionally initialize all weights to 0.5
    Matrix[] initMatrices = new Matrix[2];
    initMatrices[0] = new DenseMatrix(3, 3);
    initMatrices[0].assign(0.5);
    initMatrices[1] = new DenseMatrix(1, 4);
    initMatrices[1].assign(0.5);
    ann3.setWeightMatrices(initMatrices);

    double[] instance = {0, 1};
    Vector output = ann3.getOutput(new DenseVector(instance));
    assertEquals(0.8315410, output.get(0), 0.000001);
  }

  @Test
  public void testNeuralNetwork() throws IOException {
    testNeuralNetwork("testNeuralNetworkXORLocal", false, false, 10000);
    testNeuralNetwork("testNeuralNetworkXORWithMomentum", true, false, 5000);
    testNeuralNetwork("testNeuralNetworkXORWithRegularization", true, true, 5000);
  }

  private void testNeuralNetwork(String modelFilename, boolean useMomentum,
                                 boolean useRegularization, int iterations) throws IOException {
    NeuralNetwork ann = new MultilayerPerceptron();
    ann.addLayer(2, false, "Sigmoid");
    ann.addLayer(3, false, "Sigmoid");
    ann.addLayer(1, true, "Sigmoid");
    ann.setCostFunction("Minus_Squared").setLearningRate(0.1);

    if (useMomentum) {
      ann.setMomentumWeight(0.6);
    }

    if (useRegularization) {
      ann.setRegularizationWeight(0.01);
    }

    double[][] instances = {{0, 1, 1}, {0, 0, 0}, {1, 0, 1}, {1, 1, 0}};
    for (int i = 0; i < iterations; ++i) {
      for (double[] instance : instances) {
        ann.trainOnline(new DenseVector(instance));
      }
    }

    for (double[] instance : instances) {
      Vector input = new DenseVector(instance).viewPart(0, instance.length - 1);
      // the expected output is the last element in array
      double actual = instance[2];
      double expected = ann.getOutput(input).get(0);
      assertTrue(actual < 0.5 && expected < 0.5 || actual >= 0.5 && expected >= 0.5);
    }

    // write model into file and read out
    File tmpModelFile = this.getTestTempFile(modelFilename);
    ann.setModelPath(tmpModelFile.getAbsolutePath());
    ann.writeModelToFile();

    NeuralNetwork annCopy = new MultilayerPerceptron(tmpModelFile.getAbsolutePath());
    // test on instances
    for (double[] instance : instances) {
      Vector input = new DenseVector(instance).viewPart(0, instance.length - 1);
      // the expected output is the last element in array
      double actual = instance[2];
      double expected = annCopy.getOutput(input).get(0);
      assertTrue(actual < 0.5 && expected < 0.5 || actual >= 0.5 && expected >= 0.5);
    }
  }

  @Test
  public void testWithCancerDataSet() throws IOException {
    String dataSetPath = "src/test/resources/cancer.csv";
    List<Vector> records = Lists.newArrayList();
    // Returns a mutable list of the data
    List<String> cancerDataSetList = Files.readLines(new File(dataSetPath), Charsets.UTF_8);
    // skip the header line, hence remove the first element in the list
    cancerDataSetList.remove(0);
    for (String line : cancerDataSetList) {
      String[] tokens = CSVUtils.parseLine(line);
      double[] values = new double[tokens.length];
      for (int i = 0; i < tokens.length; ++i) {
        values[i] = Double.parseDouble(tokens[i]);
      }
      records.add(new DenseVector(values));
    }

    int splitPoint = (int) (records.size() * 0.8);
    List<Vector> trainingSet = records.subList(0, splitPoint);
    List<Vector> testSet = records.subList(splitPoint, records.size());

    // initialize neural network model
    NeuralNetwork ann = new MultilayerPerceptron();
    int featureDimension = records.get(0).size() - 1;
    ann.addLayer(featureDimension, false, "Sigmoid");
    ann.addLayer(featureDimension * 2, false, "Sigmoid");
    ann.addLayer(1, true, "Sigmoid");
    ann.setLearningRate(0.05).setMomentumWeight(0.5).setRegularizationWeight(0.001);

    int iteration = 2000;
    for (int i = 0; i < iteration; ++i) {
      for (Vector trainingInstance : trainingSet) {
        ann.trainOnline(trainingInstance);
      }
    }

    int correctInstances = 0;
    for (Vector testInstance : testSet) {
      Vector res = ann.getOutput(testInstance.viewPart(0, testInstance.size() - 1));
      double actual = res.get(0);
      double expected = testInstance.get(testInstance.size() - 1);
      if (Math.abs(actual - expected) <= 0.1) {
        ++correctInstances;
      }
    }
    double accuracy = (double) correctInstances / testSet.size() * 100;
    assertTrue("The classifier is even worse than a random guesser!", accuracy > 50);
    System.out.printf("Cancer DataSet. Classification precision: %d/%d = %f%%\n", correctInstances, testSet.size(), accuracy);
  }

  @Test
  public void testWithIrisDataSet() throws IOException {
    String dataSetPath = "src/test/resources/iris.csv";
    int numOfClasses = 3;
    List<Vector> records = Lists.newArrayList();
    // Returns a mutable list of the data
    List<String> irisDataSetList = Files.readLines(new File(dataSetPath), Charsets.UTF_8);
    // skip the header line, hence remove the first element in the list
    irisDataSetList.remove(0);

    for (String line : irisDataSetList) {
      String[] tokens = CSVUtils.parseLine(line);
      // last three dimensions represent the labels
      double[] values = new double[tokens.length + numOfClasses - 1];
      Arrays.fill(values, 0.0);
      for (int i = 0; i < tokens.length - 1; ++i) {
        values[i] = Double.parseDouble(tokens[i]);
      }
      // add label values
      String label = tokens[tokens.length - 1];
      if (label.equalsIgnoreCase("setosa")) {
        values[values.length - 3] = 1;
      } else if (label.equalsIgnoreCase("versicolor")) {
        values[values.length - 2] = 1;
      } else { // label 'virginica'
        values[values.length - 1] = 1;
      }
      records.add(new DenseVector(values));
    }

    Collections.shuffle(records);

    int splitPoint = (int) (records.size() * 0.8);
    List<Vector> trainingSet = records.subList(0, splitPoint);
    List<Vector> testSet = records.subList(splitPoint, records.size());

    // initialize neural network model
    NeuralNetwork ann = new MultilayerPerceptron();
    int featureDimension = records.get(0).size() - numOfClasses;
    ann.addLayer(featureDimension, false, "Sigmoid");
    ann.addLayer(featureDimension * 2, false, "Sigmoid");
    ann.addLayer(3, true, "Sigmoid"); // 3-class classification
    ann.setLearningRate(0.05).setMomentumWeight(0.4).setRegularizationWeight(0.005);

    int iteration = 2000;
    for (int i = 0; i < iteration; ++i) {
      for (Vector trainingInstance : trainingSet) {
        ann.trainOnline(trainingInstance);
      }
    }

    int correctInstances = 0;
    for (Vector testInstance : testSet) {
      Vector res = ann.getOutput(testInstance.viewPart(0, testInstance.size() - numOfClasses));
      double[] actualLabels = new double[numOfClasses];
      for (int i = 0; i < numOfClasses; ++i) {
        actualLabels[i] = res.get(i);
      }
      double[] expectedLabels = new double[numOfClasses];
      for (int i = 0; i < numOfClasses; ++i) {
        expectedLabels[i] = testInstance.get(testInstance.size() - numOfClasses + i);
      }

      boolean allCorrect = true;
      for (int i = 0; i < numOfClasses; ++i) {
        if (Math.abs(expectedLabels[i] - actualLabels[i]) >= 0.1) {
          allCorrect = false;
          break;
        }
      }
      if (allCorrect) {
        ++correctInstances;
      }
    }
    
    double accuracy = (double) correctInstances / testSet.size() * 100;
    assertTrue("The model is even worse than a random guesser.", accuracy > 50);
    
    System.out.printf("Iris DataSet. Classification precision: %d/%d = %f%%\n", correctInstances, testSet.size(), accuracy);
  }
  
}
