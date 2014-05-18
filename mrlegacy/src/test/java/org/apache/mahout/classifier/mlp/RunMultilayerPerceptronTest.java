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

public class RunMultilayerPerceptronTest extends MahoutTestCase {
  
  @Test
  public void runMultilayerPerceptron() throws Exception {
    
    // Train a model first
    String modelFileName = "mlp.model";
    File modelFile = getTestTempFile(modelFileName);

    File irisDataset = getTestTempFile("iris.csv");
    writeLines(irisDataset, Datasets.IRIS);

    String[] argsTrain = {
      "-i", irisDataset.getAbsolutePath(),
      "-sh",
      "-labels", "setosa", "versicolor", "virginica",
      "-mo", modelFile.getAbsolutePath(), 
      "-u",
      "-ls", "4", "8", "3"
    };
    
    TrainMultilayerPerceptron.main(argsTrain);
    
    assertTrue(modelFile.exists());
    
    String outputFileName = "labelResult.txt";
    File outputFile = getTestTempFile(outputFileName);
    
    String[] argsLabeling = {
        "-i", irisDataset.getAbsolutePath(),
        "-sh",
        "-cr", "0", "3",
        "-mo", modelFile.getAbsolutePath(),
        "-o", outputFile.getAbsolutePath()
    };
    
    RunMultilayerPerceptron.main(argsLabeling);
    
    assertTrue(outputFile.exists());
  }

}
