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

package org.apache.mahout.classifier.naivebayes;

import org.apache.mahout.math.DenseVector;
import org.junit.Before;
import org.junit.Test;


public final class StandardNaiveBayesClassifierTest extends NaiveBayesTestBase {

  private StandardNaiveBayesClassifier classifier;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    NaiveBayesModel model = createNaiveBayesModel();
    classifier = new StandardNaiveBayesClassifier(model);
  }
  
  @Test
  public void testNaiveBayes() throws Exception {
    assertEquals(4, classifier.numCategories());
    assertEquals(0, maxIndex(classifier.classifyFull(new DenseVector(new double[] { 1.0, 0.0, 0.0, 0.0 }))));
    assertEquals(1, maxIndex(classifier.classifyFull(new DenseVector(new double[] { 0.0, 1.0, 0.0, 0.0 }))));
    assertEquals(2, maxIndex(classifier.classifyFull(new DenseVector(new double[] { 0.0, 0.0, 1.0, 0.0 }))));
    assertEquals(3, maxIndex(classifier.classifyFull(new DenseVector(new double[] { 0.0, 0.0, 0.0, 1.0 }))));
    
  }
  
}
