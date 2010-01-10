package org.apache.mahout.classifier.discriminative;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import junit.framework.TestCase;

public class PerceptronTrainerTest extends TestCase {

  private PerceptronTrainer trainer;

  protected void setUp() throws Exception {
    super.setUp();
    trainer = new PerceptronTrainer(3, 0.5, 0.1, 1.0, 1.0);
  }

  public void testUpdate() throws TrainingException {
    double[] labels = { 1.0, 1.0, 1.0, 0.0 };
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
    this.trainer.train(labelset, dataset);
    assertFalse(this.trainer.getModel().classify(dataset.getColumn(3)));
    assertTrue(this.trainer.getModel().classify(dataset.getColumn(0)));
  }

}
