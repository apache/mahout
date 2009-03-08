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

package org.apache.mahout.classifier.bayes;

import junit.framework.TestCase;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.cbayes.CBayesModel;

public class CBayesClassifierTest extends TestCase {

  protected CBayesModel model;

  public CBayesClassifierTest(String s) {
    super(s);
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    model = new CBayesModel();
    //String[] labels = new String[]{"a", "b", "c", "d", "e"};
    //long[] labelCounts = new long[]{6, 20, 60, 100, 200};
    //String[] features = new String[]{"aa", "bb", "cc", "dd", "ee"};
    model.setSigma_jSigma_k(500.0);

    model.setSumFeatureWeight("aa", 80);
    model.setSumFeatureWeight("bb", 21);
    model.setSumFeatureWeight("cc", 60);
    model.setSumFeatureWeight("dd", 115);
    model.setSumFeatureWeight("ee", 100);

    model.setSumLabelWeight("a", 100);
    model.setSumLabelWeight("b", 100);
    model.setSumLabelWeight("c", 100);
    model.setSumLabelWeight("d", 100);
    model.setSumLabelWeight("e", 100);

    model.setThetaNormalizer("a", -100);
    model.setThetaNormalizer("b", -100);
    model.setThetaNormalizer("c", -100);
    model.setThetaNormalizer("d", -100);
    model.setThetaNormalizer("e", -100);

    model.initializeNormalizer();
    model.initializeWeightMatrix();

    model.loadFeatureWeight("a", "aa", 5);
    model.loadFeatureWeight("a", "bb", 1);

    model.loadFeatureWeight("b", "bb", 20);

    model.loadFeatureWeight("c", "cc", 30);
    model.loadFeatureWeight("c", "aa", 25);
    model.loadFeatureWeight("c", "dd", 5);

    model.loadFeatureWeight("d", "dd", 60);
    model.loadFeatureWeight("d", "cc", 40);

    model.loadFeatureWeight("e", "ee", 100);
    model.loadFeatureWeight("e", "aa", 50);
    model.loadFeatureWeight("e", "dd", 50);
  }

  public void test() {
    BayesClassifier classifier = new BayesClassifier();
    String[] document = {"aa", "ff"};
    ClassifierResult result = classifier.classify(model, document, "unknown");
    assertNotNull("category is null and it shouldn't be", result);
    assertEquals(result + " is not equal to e", "e", result.getLabel());

    document = new String[]{"ff"};
    result = classifier.classify(model, document, "unknown");
    assertNotNull("category is null and it shouldn't be", result);
    assertEquals(result + " is not equal to d", "d", result.getLabel());

    document = new String[]{"cc"};
    result = classifier.classify(model, document, "unknown");
    assertNotNull("category is null and it shouldn't be", result);
    assertEquals(result + " is not equal to d", "d", result.getLabel());
  }

  public void testResults() throws Exception {
    BayesClassifier classifier = new BayesClassifier();
    String[] document = {"aa", "ff"};
    ClassifierResult result = classifier.classify(model, document, "unknown");
    assertNotNull("category is null and it shouldn't be", result);
    System.out.println("Result: " + result);
  }
}