package org.apache.mahout.classifier.bayes;

/**
 * Copyright 2004 The Apache Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import junit.framework.TestCase;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.common.Model;

public class BayesClassifierTest extends TestCase {
  protected Model model;


  public BayesClassifierTest(String s) {
    super(s);
  }

  protected void setUp() {
    model = new BayesModel();
    //String[] labels = new String[]{"a", "b", "c", "d", "e"};
    //long[] labelCounts = new long[]{6, 20, 60, 100, 200};
    //String[] features = new String[]{"aa", "bb", "cc", "dd", "ee"};
    model.setSigma_jSigma_k(100.0f);
    
    model.setSumFeatureWeight("aa", 100);
    model.setSumFeatureWeight("bb", 100);
    model.setSumFeatureWeight("cc", 100);
    model.setSumFeatureWeight("dd", 100);
    model.setSumFeatureWeight("ee", 100);
    
    model.setSumLabelWeight("a", 1);
    model.setSumLabelWeight("b", 1);
    model.setSumLabelWeight("c", 1);
    model.setSumLabelWeight("d", 1);
    model.setSumLabelWeight("e", 1);
    
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

  protected void tearDown() {

  }

  public void test() {
    BayesClassifier classifier = new BayesClassifier();
    ClassifierResult result;
    String[] document = new String[]{"aa", "ff"};
    result = classifier.classify(model, document, "unknown");
    assertTrue("category is null and it shouldn't be", result != null);
    assertTrue(result + " is not equal to " + "e", result.getLabel().equals("e") == true);

    document = new String[]{"ff"};
    result = classifier.classify(model, document, "unknown");
    assertTrue("category is null and it shouldn't be", result != null);
    assertTrue(result + " is not equal to " + "unknown", result.getLabel().equals("unknown") == true);

    document = new String[]{"cc"};
    result = classifier.classify(model, document, "unknown");
    assertTrue("category is null and it shouldn't be", result != null);
    assertTrue(result + " is not equal to " + "d", result.getLabel().equals("d") == true);
  }

  public void testResults() throws Exception {
    BayesClassifier classifier = new BayesClassifier();
    String[] document = new String[]{"aa", "ff"};
    ClassifierResult result = classifier.classify(model, document, "unknown");
    assertTrue("category is null and it shouldn't be", result != null);    
    System.out.println("Result: " + result);
  }
}