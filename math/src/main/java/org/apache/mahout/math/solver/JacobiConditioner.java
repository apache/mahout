package org.apache.mahout.math.solver;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * 
 * Implements the Jacobi preconditioner for a matrix A. This is defined as inv(diag(A)).
 *
 */
public class JacobiConditioner implements Preconditioner
{
  private DenseVector inverseDiagonal;

  public JacobiConditioner(Matrix a) {
    if (a.numCols() != a.numRows()) {
      throw new IllegalArgumentException("Matrix must be square.");
    }
    
    inverseDiagonal = new DenseVector(a.numCols());
    for (int i = 0; i < a.numCols(); ++i) {
      inverseDiagonal.setQuick(i, 1.0 / a.getQuick(i, i));
    }
  }
  
  @Override
  public Vector precondition(Vector v)
  {
    return v.times(inverseDiagonal);
  }

}
