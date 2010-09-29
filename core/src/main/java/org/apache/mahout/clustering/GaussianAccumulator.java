package org.apache.mahout.clustering;

import org.apache.mahout.math.Vector;

public interface GaussianAccumulator {

  /**
   * @return the number of observations
   */
  public abstract double getN();

  /**
   * @return the mean of the observations
   */
  public abstract Vector getMean();

  /**
   * @return the std of the observations
   */
  public abstract Vector getStd();
  
  /**
   * @return the average of the vector std elements
   */
  public abstract double getAverageStd();
  
  /**
   * @return the variance of the observations
   */
  public abstract Vector getVariance();

  /**
   * Observe the vector with the given weight
   * 
   * @param x a Vector
   * @param weight a double
   */
  public abstract void observe(Vector x, double weight);

  /**
   * Compute the mean and standard deviation
   */
  public abstract void compute();

}
