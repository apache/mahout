package org.apache.mahout.classifier.naivebayes;

import org.apache.mahout.math.DenseVector;
import org.junit.Before;
import org.junit.Test;


public final class ComplementaryNaiveBayesClassifierTest extends NaiveBayesTestBase{

  NaiveBayesModel model;
  ComplementaryNaiveBayesClassifier classifier;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    model = createComplementaryNaiveBayesModel();
    classifier = new ComplementaryNaiveBayesClassifier(model);
  }
  
  @Test
  public void testNaiveBayes() throws Exception {
    assertEquals(classifier.numCategories(), 4);
    assertEquals(0, maxIndex(classifier.classify(new DenseVector(new double[] {1.0, 0.0, 0.0, 0.0}))));
    assertEquals(1, maxIndex(classifier.classify(new DenseVector(new double[] {0.0, 1.0, 0.0, 0.0}))));
    assertEquals(2, maxIndex(classifier.classify(new DenseVector(new double[] {0.0, 0.0, 1.0, 0.0}))));
    assertEquals(3, maxIndex(classifier.classify(new DenseVector(new double[] {0.0, 0.0, 0.0, 1.0}))));
    
  }
  
}
