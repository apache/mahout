package org.apache.mahout.ep;

import org.apache.mahout.math.function.UnaryFunction;

/**
 * Provides coordinate tranformations so that evolution can proceed on the entire space of
 * reals but have the output limited and squished in convenient (and safe) ways.
 */
public abstract class Mapping implements UnaryFunction {
  public static class SoftLimit extends Mapping {
    private double min;
    private double max;
    private double scale;

    @SuppressWarnings({"UnusedDeclaration"})
    private SoftLimit() {
    }

    private SoftLimit(double min, double max, double scale) {
      this.min = min;
      this.max = max;
      this.scale = scale;
    }

    public double apply(double v) {
      return min + (max - min) * 1 / (1 + Math.exp(-v * scale));
    }

  }

  public static class LogLimit extends Mapping {
    private Mapping wrapped;

    @SuppressWarnings({"UnusedDeclaration"})
    private LogLimit() {
    }

    private LogLimit(double low, double high) {
      wrapped = softLimit(Math.log(low), Math.log(high));
    }

    @Override
    public double apply(double v) {
      return Math.exp(wrapped.apply(v));
    }
  }

  public static class Exponential extends Mapping {
    private double scale;

    @SuppressWarnings({"UnusedDeclaration"})
    private Exponential() {
    }

    private Exponential(double scale) {
      this.scale = scale;
    }

    @Override
    public double apply(double v) {
      return Math.exp(v * scale);
    }
  }

  public static class Identity extends Mapping {
    private Identity() {
    }

    @Override
    public double apply(double v) {
      return v;
    }
  }

  /**
   * Maps input to the open interval (min, max) with 0 going to the mean of min and
   * max.  When scale is large, a larger proportion of values are mapped to points
   * near the boundaries.  When scale is small, a larger proportion of values are mapped to
   * points well within the boundaries.
   * @param min The largest lower bound on values to be returned.
   * @param max The least upper bound on values to be returned.
   * @param scale  Defines how sharp the boundaries are.
   * @return A mapping that satisfies the desired constraint.
   */
  public static Mapping softLimit(final double min, final double max, final double scale) {
    return new SoftLimit(min, max, scale);
  }

  /**
   * Maps input to the open interval (min, max) with 0 going to the mean of min and
   * max.  When scale is large, a larger proportion of values are mapped to points
   * near the boundaries.
   * @see #softLimit(double, double, double)
   * @param min The largest lower bound on values to be returned.
   * @param max The least upper bound on values to be returned.
   * @return A mapping that satisfies the desired constraint.
   */
  public static Mapping softLimit(double min, double max) {
    return softLimit(min, max, 1);
  }

  /**
   * Maps input to positive values in the open interval (min, max) with
   * 0 going to the geometric mean.  Near the geometric mean, values are
   * distributed roughly geometrically.
   * @param low   The largest lower bound for output results.  Must be >0.
   * @param high  The least upper bound for output results.  Must be >0.
   * @return A mapped value.
   */
  public static Mapping logLimit(final double low, final double high) {
    if (low <= 0) {
      throw new IllegalArgumentException("Lower bound for log limit must be > 0 but was " + low);
    }
    if (high <= 0) {
      throw new IllegalArgumentException("Upper bound for log limit must be > 0 but was " + high);
    }
    return new LogLimit(low, high);
  }

  /**
   * Maps results to positive values.
   * @return A positive value.
   */
  public static Mapping exponential() {
    return exponential(1);
  }

  /**
   * Maps results to positive values.
   * @param scale  If large, then large values are more likely.
   * @return A positive value.
   */
  public static Mapping exponential(final double scale) {
    return new Exponential(scale);
  }

  /**
   * Maps results to themselves.
   * @return The original value.
   */
  public static Mapping identity() {
    return new Identity();
  }
}
