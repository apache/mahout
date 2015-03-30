/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public class TrainMultilayerPerceptronTest extends MahoutTestCase {
  
  @Test
  public void testIrisDataset() throws Exception {
    String modelFileName = "mlp.model";
    File modelFile = getTestTempFile(modelFileName);

    File irisDataset = getTestTempFile("iris.csv");
    writeLines(irisDataset, Datasets.IRIS);

    String[] args = {
      "-i", irisDataset.getAbsolutePath(),
      "-sh",
      "-labels", "setosa", "versicolor", "virginica",
      "-mo", modelFile.getAbsolutePath(), 
      "-u",
      "-ls", "4", "8", "3"
    };
    
    TrainMultilayerPerceptron.main(args);
    
    assertTrue(modelFile.exists());
  }
  
  @Test
  public void initializeModelWithDifferentParameters() throws Exception {
    String modelFileName = "mlp.model";
    File modelFile1 = getTestTempFile(modelFileName);

    File irisDataset = getTestTempFile("iris.csv");
    writeLines(irisDataset, Datasets.IRIS);

    String[] args1 = {
        "-i", irisDataset.getAbsolutePath(),
        "-sh",
        "-labels", "setosa", "versicolor", "virginica",
        "-mo", modelFile1.getAbsolutePath(), 
        "-u",
        "-ls", "4", "8", "3",
        "-l", "0.2", "-m", "0.35", "-r", "0.0001"
      };
    
    MultilayerPerceptron mlp1 = trainModel(args1, modelFile1);
    assertEquals(0.2, mlp1.getLearningRate(), EPSILON);
    assertEquals(0.35, mlp1.getMomentumWeight(), EPSILON);
    assertEquals(0.0001, mlp1.getRegularizationWeight(), EPSILON);
    
    assertEquals(4, mlp1.getLayerSize(0) - 1);
    assertEquals(8, mlp1.getLayerSize(1) - 1);
    assertEquals(3, mlp1.getLayerSize(2));  // Final layer has no bias neuron
    
    // MLP with default learning rate, momemtum weight, and regularization weight
    File modelFile2 = this.getTestTempFile(modelFileName);
    
    String[] args2 = {
        "-i", irisDataset.getAbsolutePath(),
        "-sh",
        "-labels", "setosa", "versicolor", "virginica",
        "-mo", modelFile2.getAbsolutePath(), 
        "-ls", "4", "10", "18", "3"
      };
    
    MultilayerPerceptron mlp2 = trainModel(args2, modelFile2);
    assertEquals(0.5, mlp2.getLearningRate(), EPSILON);
    assertEquals(0.1, mlp2.getMomentumWeight(), EPSILON);
    assertEquals(0, mlp2.getRegularizationWeight(), EPSILON);
    
    assertEquals(4, mlp2.getLayerSize(0) - 1);
    assertEquals(10, mlp2.getLayerSize(1) - 1);
    assertEquals(18, mlp2.getLayerSize(2) - 1);
    assertEquals(3, mlp2.getLayerSize(3));  // Final layer has no bias neuron
    
  }
  
  private MultilayerPerceptron trainModel(String[] args, File modelFile) throws Exception {
    TrainMultilayerPerceptron.main(args);
    return new MultilayerPerceptron(modelFile.getAbsolutePath());
  }

}
