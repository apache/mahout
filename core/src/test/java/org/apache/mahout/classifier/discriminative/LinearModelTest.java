package org.apache.mahout.classifier.discriminative;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import junit.framework.TestCase;

public class LinearModelTest extends TestCase {

  private LinearModel model;
  private Vector hyperplane;

  protected void setUp() throws Exception {
    super.setUp();
    double[] values = {0.0, 1.0, 0.0, 1.0, 0.0};
    this.hyperplane = new DenseVector(values);
    this.model = new LinearModel(this.hyperplane, 0.1, 0.5);
  }

  public void testClassify() {
    double[] valuesFalse = {1.0, 0.0, 1.0, 0.0, 1.0};
    Vector dataPointFalse = new DenseVector(valuesFalse);
    assertFalse(this.model.classify(dataPointFalse));

    double[] valuesTrue = {0.0, 1.0, 0.0, 1.0, 0.0};
    Vector dataPointTrue = new DenseVector(valuesTrue);
    assertTrue(this.model.classify(dataPointTrue));
  }

  public void testAddDelta() {
    double[] values = {1.0, -1.0, 1.0, -1.0, 1.0};
    this.model.addDelta(new DenseVector(values));

    double[] valuesFalse = {1.0, 0.0, 1.0, 0.0, 1.0};
    Vector dataPointFalse = new DenseVector(valuesFalse);
    assertTrue(this.model.classify(dataPointFalse));

    double[] valuesTrue = {0.0, 1.0, 0.0, 1.0, 0.0};
    Vector dataPointTrue = new DenseVector(valuesTrue);
    assertFalse(this.model.classify(dataPointTrue));
  }

  public void testTimesDelta() {
    double[] values = {-1.0, -1.0, -1.0, -1.0, -1.0};
    this.model.addDelta(new DenseVector(values));
    double[] dotval = {-1.0, -1.0, -1.0, -1.0, -1.0};
    
    for (int i = 0; i < dotval.length; i++) {
      this.model.timesDelta(i, dotval[i]);
    }

    double[] valuesFalse = {1.0, 0.0, 1.0, 0.0, 1.0};
    Vector dataPointFalse = new DenseVector(valuesFalse);
    assertTrue(this.model.classify(dataPointFalse));

    double[] valuesTrue = {0.0, 1.0, 0.0, 1.0, 0.0};
    Vector dataPointTrue = new DenseVector(valuesTrue);
    assertFalse(this.model.classify(dataPointTrue));
  }

}
