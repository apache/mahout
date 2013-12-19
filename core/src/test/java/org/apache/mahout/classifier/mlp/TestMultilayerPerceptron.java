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
package org.apache.mahout.classifier.mlp;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Arrays;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

/**
 * Test the functionality of {@link MultilayerPerceptron}
 */
public class TestMultilayerPerceptron extends MahoutTestCase {

  @Test
  public void testMLP() throws IOException {
    testMLP("testMLPXORLocal", false, false, 8000);
    testMLP("testMLPXORLocalWithMomentum", true, false, 4000);
    testMLP("testMLPXORLocalWithRegularization", true, true, 2000);
  }

  private void testMLP(String modelFilename, boolean useMomentum,
      boolean useRegularization, int iterations) throws IOException {
    MultilayerPerceptron mlp = new MultilayerPerceptron();
    mlp.addLayer(2, false, "Sigmoid");
    mlp.addLayer(3, false, "Sigmoid");
    mlp.addLayer(1, true, "Sigmoid");
    mlp.setCostFunction("Minus_Squared").setLearningRate(0.2);
    if (useMomentum) {
      mlp.setMomentumWeight(0.6);
    }

    if (useRegularization) {
      mlp.setRegularizationWeight(0.01);
    }

    double[][] instances = { { 0, 1, 1 }, { 0, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 } };
    for (int i = 0; i < iterations; ++i) {
      for (double[] instance : instances) {
        Vector features = new DenseVector(Arrays.copyOf(instance, instance.length - 1));
        mlp.train((int) instance[2], features);
      }
    }

    for (double[] instance : instances) {
      Vector input = new DenseVector(instance).viewPart(0, instance.length - 1);
      // the expected output is the last element in array
      double actual = instance[2];
      double expected = mlp.getOutput(input).get(0);
      assertTrue(actual < 0.5 && expected < 0.5 || actual >= 0.5 && expected >= 0.5);
    }

    // write model into file and read out
    File modelFile = this.getTestTempFile(modelFilename);
    mlp.setModelPath(modelFile.getAbsolutePath());
    mlp.writeModelToFile();
    mlp.close();

    MultilayerPerceptron mlpCopy = new MultilayerPerceptron(modelFile.getAbsolutePath());
    // test on instances
    for (double[] instance : instances) {
      Vector input = new DenseVector(instance).viewPart(0, instance.length - 1);
      // the expected output is the last element in array
      double actual = instance[2];
      double expected = mlpCopy.getOutput(input).get(0);
      assertTrue(actual < 0.5 && expected < 0.5 || actual >= 0.5 && expected >= 0.5);
    }
    mlpCopy.close();
  }
}
