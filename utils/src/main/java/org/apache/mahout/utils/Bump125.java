package org.apache.mahout.utils;

/**
 * Helps with making nice intervals at arbitrary scale.
 *
 * One use case is where we are producing progress or error messages every time an incoming
 * record is received.  It is generally bad form to produce a message for <i>every</i> input
 * so it would be better to produce a message for each of the first 10 records, then every
 * other record up to 20 and then every 5 records up to 50 and then every 10 records up to 100,
 * more or less. The pattern can now repeat scaled up by 100.  The total number of messages will scale
 * with the log of the number of input lines which is much more survivable than direct output
 * and because early records all get messages, we get indications early.
 */
public class Bump125 {
  private static final int[] bumps = {1, 2, 5};

  static int scale(double value, double base) {
    double scale = value / base;
    // scan for correct step
    int i = 0;
    while (i < bumps.length - 1 && bumps[i + 1] <= scale) {
      i++;
    }
    return bumps[i];
  }

  static long base(double value) {
    return Math.max(1, (long) Math.pow(10, (int) Math.floor(Math.log10(value))));
  }

  private long counter = 0;

  public long increment() {
    long delta;
    if (counter >= 10) {
      final long base = base(counter / 4.0);
      int scale = scale(counter / 4.0, base);
      delta = (long) (base * scale);
    } else {
      delta = 1;
    }
    counter += delta;
    return counter;
  }
}
