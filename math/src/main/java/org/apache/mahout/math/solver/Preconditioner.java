package org.apache.mahout.math.solver;

import org.apache.mahout.math.Vector;

/**
 * 
 * <p>Interface for defining preconditioners used for improving the performance and/or stability of linear
 * system solvers.
 *
 */
public interface Preconditioner
{
  /**
   * Preconditions the specified vector.
   * 
   * @param v The vector to precondition.
   * @return The preconditioned vector.
   */
  public Vector precondition(Vector v);
}
