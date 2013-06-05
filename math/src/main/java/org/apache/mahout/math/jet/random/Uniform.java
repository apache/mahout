/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.common.RandomUtils;

import java.util.Random;

public class Uniform extends AbstractContinousDistribution {

  private double min;
  private double max;

  /**
   * Constructs a uniform distribution with the given minimum and maximum, using a {@link
   * org.apache.mahout.math.jet.random.engine.MersenneTwister} seeded with the given seed.
   */
  public Uniform(double min, double max, int seed) {
    this(min, max, RandomUtils.getRandom(seed));
  }

  /** Constructs a uniform distribution with the given minimum and maximum. */
  public Uniform(double min, double max, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(min, max);
  }

  /** Constructs a uniform distribution with <tt>min=0.0</tt> and <tt>max=1.0</tt>. */
  public Uniform(Random randomGenerator) {
    this(0, 1, randomGenerator);
  }

  /** Returns the cumulative distribution function (assuming a continous uniform distribution). */
  @Override
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
    return randomDouble() > 0.5;
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(min,max)</tt> (excluding <tt>min</tt> and
   * <tt>max</tt>).
   */
  @Override
  public double nextDouble() {
    return min + (max - min) * randomDouble();
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(from,to)</tt> (excluding <tt>from</tt> and
   * <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public double nextDoubleFromTo(double from, double to) {
    return from + (to - from) * randomDouble();
  }

  /**
   * Returns a uniformly distributed random number in the open interval <tt>(from,to)</tt> (excluding <tt>from</tt> and
   * <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public float nextFloatFromTo(float from, float to) {
    return (float) nextDoubleFromTo(from, to);
  }

  /**
   * Returns a uniformly distributed random number in the closed interval
   *  <tt>[from,to]</tt> (including <tt>from</tt>
   * and <tt>to</tt>). Pre conditions: <tt>from &lt;= to</tt>.
   */
  public int nextIntFromTo(int from, int to) {
    return (int) (from + (long) ((1L + to - from) * randomDouble()));
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
      return from + (long) nextDoubleFromTo(0.0, to - from + 1);
    }

    // would we get a numeric overflow?
    // if not, we can still handle the case rather efficient.
    double diff = (double) to - (double) from + 1.0;
    if (diff <= Long.MAX_VALUE) {
      return from + (long) nextDoubleFromTo(0.0, diff);
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
      random = Math.round(nextDoubleFromTo(Long.MIN_VALUE, to + 1));
      if (random > to) {
        random = Long.MIN_VALUE;
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
  @Override
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


  /** Returns a String representation of the receiver. */
  @Override
  public String toString() {
    return this.getClass().getName() + '(' + min + ',' + max + ')';
  }
}
