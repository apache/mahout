/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.math.jet.random.engine.MersenneTwister;
import org.apache.mahout.math.jet.random.engine.RandomEngine;

public class Uniform extends AbstractContinousDistribution {

  private double min;
  private double max;

  // The uniform random number generated shared by all <b>static</b> methods.
  protected static final Uniform shared = new Uniform(makeDefaultGenerator());

  /**
   * Constructs a uniform distribution with the given minimum and maximum, using a {@link
   * org.apache.mahout.math.jet.random.engine.MersenneTwister} seeded with the given seed.
   */
  public Uniform(double min, double max, int seed) {
    this(min, max, new MersenneTwister(seed));
  }

  /** Constructs a uniform distribution with the given minimum and maximum. */
  public Uniform(double min, double max, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(min, max);
  }

  /** Constructs a uniform distribution with <tt>min=0.0</tt> and <tt>max=1.0</tt>. */
  public Uniform(RandomEngine randomGenerator) {
    this(0, 1, randomGenerator);
  }

  /** Returns the cumulative distribution function (assuming a continous uniform distribution). */
  public double cdf(double x) {
    if (x <= min) {
      return 0.0;
    }
    if (x >= max) {
      return 1.0;
    }
    return (x - min) / (max - min);
  }

  /** Returns a uniformly distributed random <tt>boolean</tt>. */
  public boolean nextBoolean() {
    return randomGenerator.raw() > 0.5;
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(min,max)</tt> (excluding <tt>min</tt> and
   * <tt>max</tt>).
   */
  @Override
  public double nextDouble() {
    return min + (max - min) * randomGenerator.raw();
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(from,to)</tt> (excluding <tt>from</tt> and
   * <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public double nextDoubleFromTo(double from, double to) {
    return from + (to - from) * randomGenerator.raw();
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(from,to)</tt> (excluding <tt>from</tt> and
   * <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public float nextFloatFromTo(float from, float to) {
    return (float) nextDoubleFromTo(from, to);
  }

  /**
   * Returns a uniformly distributed random number in the closed interval <tt>[min,max]</tt> (including <tt>min</tt> and
   * <tt>max</tt>).
   */
  @Override
  public int nextInt() {
    return nextIntFromTo((int) Math.round(min), (int) Math.round(max));
  }

  /**
   * Returns a uniformly distributed random number in the closed interval
   *  <tt>[from,to]</tt> (including <tt>from</tt>
   * and <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public int nextIntFromTo(int from, int to) {
    return (int) ((long) from + (long) ((1L + (long) to - (long) from) * randomGenerator.raw()));
  }

  /**
   * Returns a uniformly distributed random number in the closed interval <tt>[from,to]</tt> (including <tt>from</tt>
   * and <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public long nextLongFromTo(long from, long to) {
    /* Doing the thing turns out to be more tricky than expected.
       avoids overflows and underflows.
       treats cases like from=-1, to=1 and the like right.
       the following code would NOT solve the problem: return (long) (Doubles.randomFromTo(from,to));

       rounding avoids the unsymmetric behaviour of casts from double to long: (long) -0.7 = 0, (long) 0.7 = 0.
       checking for overflows and underflows is also necessary.
    */

    // first the most likely and also the fastest case.
    if (from >= 0 && to < Long.MAX_VALUE) {
      return from + (long) (nextDoubleFromTo(0.0, to - from + 1));
    }

    // would we get a numeric overflow?
    // if not, we can still handle the case rather efficient.
    double diff = ((double) to) - (double) from + 1.0;
    if (diff <= Long.MAX_VALUE) {
      return from + (long) (nextDoubleFromTo(0.0, diff));
    }

    // now the pathologic boundary cases.
    // they are handled rather slow.
    long random;
    if (from == Long.MIN_VALUE) {
      if (to == Long.MAX_VALUE) {
        //return Math.round(nextDoubleFromTo(from,to));
        int i1 = nextIntFromTo(Integer.MIN_VALUE, Integer.MAX_VALUE);
        int i2 = nextIntFromTo(Integer.MIN_VALUE, Integer.MAX_VALUE);
        return ((i1 & 0xFFFFFFFFL) << 32) | (i2 & 0xFFFFFFFFL);
      }
      random = Math.round(nextDoubleFromTo(from, to + 1));
      if (random > to) {
        random = from;
      }
    } else {
      random = Math.round(nextDoubleFromTo(from - 1, to));
      if (random < from) {
        random = to;
      }
    }
    return random;
  }

  /** Returns the probability distribution function (assuming a continous uniform distribution). */
  public double pdf(double x) {
    if (x <= min || x >= max) {
      return 0.0;
    }
    return 1.0 / (max - min);
  }

  /** Sets the internal state. */
  public void setState(double min, double max) {
    if (max < min) {
      setState(max, min);
      return;
    }
    this.min = min;
    this.max = max;
  }

  /** Returns a uniformly distributed random <tt>boolean</tt>. */
  public static boolean staticNextBoolean() {
    synchronized (shared) {
      return shared.nextBoolean();
    }
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(0,1)</tt> (excluding <tt>0</tt> and
   * <tt>1</tt>).
   */
  public static double staticNextDouble() {
    synchronized (shared) {
      return shared.nextDouble();
    }
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(from,to)</tt> (excluding <tt>from</tt> and
   * <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public static double staticNextDoubleFromTo(double from, double to) {
    synchronized (shared) {
      return shared.nextDoubleFromTo(from, to);
    }
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(from,to)</tt> (excluding <tt>from</tt> and
   * <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public static float staticNextFloatFromTo(float from, float to) {
    synchronized (shared) {
      return shared.nextFloatFromTo(from, to);
    }
  }

  /**
   * Returns a uniformly distributed random number in the closed interval <tt>[from,to]</tt> (including <tt>from</tt>
   * and <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public static int staticNextIntFromTo(int from, int to) {
    synchronized (shared) {
      return shared.nextIntFromTo(from, to);
    }
  }

  /**
   * Returns a uniformly distributed random number in the closed interval <tt>[from,to]</tt> (including <tt>from</tt>
   * and <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public static long staticNextLongFromTo(long from, long to) {
    synchronized (shared) {
      return shared.nextLongFromTo(from, to);
    }
  }

  /**
   * Sets the uniform random number generation engine shared by all <b>static</b> methods.
   *
   * @param randomGenerator the new uniform random number generation engine to be shared.
   */
  public static void staticSetRandomEngine(RandomEngine randomGenerator) {
    synchronized (shared) {
      shared.setRandomGenerator(randomGenerator);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + min + ',' + max + ')';
  }
}
