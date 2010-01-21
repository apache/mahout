/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A handy stopwatch for benchmarking.
 * Like a real stop watch used on ancient running tracks you can start the watch, stop it,
 * start it again, stop it again, display the elapsed time and reset the watch.
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Timer extends PersistentObject {

  private static final Logger log = LoggerFactory.getLogger(Timer.class);

  private long baseTime;
  private long elapsedTime;

  private static final long UNIT = 1000;

  /** Constructs a new timer, initially not started. Use start() to start the timer. */
  public Timer() {
    this.reset();
  }

  /**
   * Prints the elapsed time on System.out
   *
   * @return <tt>this</tt> (for convenience only).
   */
  public Timer display() {
    log.info(this.toString());
    return this;
  }

  /** Same as <tt>seconds()</tt>. */
  public float elapsedTime() {
    return seconds();
  }

  /** Returns the elapsed time in milli seconds; does not stop the timer, if started. */
  public long millis() {
    long elapsed = elapsedTime;
    if (baseTime != 0) { // we are started
      elapsed += System.currentTimeMillis() - baseTime;
    }
    return elapsed;
  }

  /**
   * <tt>T = this - other</tt>; Constructs and returns a new timer which is the difference of the receiver and the other
   * timer. The new timer is not started.
   *
   * @param other the timer to subtract.
   * @return a new timer.
   */
  public Timer minus(Timer other) {
    Timer copy = new Timer();
    copy.elapsedTime = millis() - other.millis();
    return copy;
  }

  /** Returns the elapsed time in minutes; does not stop the timer, if started. */
  public float minutes() {
    return seconds() / 60;
  }

  /**
   * <tt>T = this + other</tt>; Constructs and returns a new timer which is the sum of the receiver and the other timer.
   * The new timer is not started.
   *
   * @param other the timer to add.
   * @return a new timer.
   */
  public Timer plus(Timer other) {
    Timer copy = new Timer();
    copy.elapsedTime = millis() + other.millis();
    return copy;
  }

  /**
   * Resets the timer.
   *
   * @return <tt>this</tt> (for convenience only).
   */
  public Timer reset() {
    elapsedTime = 0;
    baseTime = 0;
    return this;
  }

  /** Returns the elapsed time in seconds; does not stop the timer, if started. */
  public float seconds() {
    return ((float) millis()) / UNIT;
  }

  /**
   * Starts the timer.
   *
   * @return <tt>this</tt> (for convenience only).
   */
  public Timer start() {
    baseTime = System.currentTimeMillis();
    return this;
  }

  /**
   * Stops the timer. You can start it again later, if necessary.
   *
   * @return <tt>this</tt> (for convenience only).
   */
  public Timer stop() {
    if (baseTime != 0) {
      elapsedTime += (System.currentTimeMillis() - baseTime);
    }
    baseTime = 0;
    return this;
  }

  /** Shows how to use a timer in convenient ways. */
  public static void test(int size) {
    //benchmark this piece
    Timer t = new Timer().start();
    int j = 0;
    for (int i = 0; i < size; i++) {
      j++;
    }
    t.stop();
    t.display();


    //do something we do not want to benchmark
    j = 0;
    for (int i = 0; i < size; i++) {
      j++;
    }


    //benchmark another piece and add to last benchmark
    t.start();
    j = 0;
    for (int i = 0; i < size; i++) {
      j++;
    }
    t.stop().display();


    //benchmark yet another piece independently
    t.reset(); //set timer to zero
    t.start();
    j = 0;
    for (int i = 0; i < size; i++) {
      j++;
    }
    t.stop().display();
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return "Time=" + Float.toString(this.elapsedTime()) + " secs";
  }
}
