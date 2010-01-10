package org.apache.mahout.classifier.discriminative;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import junit.framework.TestCase;


public class WinnowTrainerTest extends TestCase {

  private WinnowTrainer trainer;

  protected void setUp() throws Exception {
    super.setUp();
    trainer = new WinnowTrainer(3);
  }

  public void testUpdate() throws Exception {
    double[] labels = { 0.0, 0.0, 0.0, 1.0 };
    Vector labelset = new DenseVector(labels);
    double[][] values = new double[3][4];
    for (int i = 0; i < 3; i++) {
      values[i][0] = 1.0;
      values[i][1] = 1.0;
      values[i][2] = 1.0;
      values[i][3] = 1.0;
    }
    values[1][0] = 0.0;
    values[2][0] = 0.0;
    values[1][1] = 0.0;
    values[2][2] = 0.0;

    Matrix dataset = new DenseMatrix(values);
    trainer.train(labelset, dataset);
    assertTrue(trainer.getModel().classify(dataset.getColumn(3)));
    assertFalse(trainer.getModel().classify(dataset.getColumn(0)));
  }

}
