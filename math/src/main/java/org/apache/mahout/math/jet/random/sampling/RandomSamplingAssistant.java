/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random.sampling;

import java.util.Random;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public final class RandomSamplingAssistant {

  private static final int MAX_BUFFER_SIZE = 200;

  //public class RandomSamplingAssistant extends Object implements java.io.Serializable {
  private final RandomSampler sampler;
  private final long[] buffer;
  private int bufferPosition;

  private long skip;
  private long n;

  /**
   * Constructs a random sampler that samples <tt>n</tt> random elements from an input sequence of <tt>N</tt> elements.
   *
   * @param n               the total number of elements to choose (must be &gt;= 0).
   * @param N               number of elements to choose from (must be &gt;= n).
   * @param randomGenerator a random number generator. Set this parameter to <tt>null</tt> to use the default random
   *                        number generator.
   */
  public RandomSamplingAssistant(long n, long N, Random randomGenerator) {
    this.n = n;
    this.sampler = new RandomSampler(n, N, 0, randomGenerator);
    this.buffer = new long[(int) Math.min(n, MAX_BUFFER_SIZE)];
    if (n > 0) {
      this.buffer[0] = -1;
    } // start with the right offset

    fetchNextBlock();
  }

  /** Not yet commented. */
  void fetchNextBlock() {
    if (n > 0) {
      long last = buffer[bufferPosition];
      sampler.nextBlock((int) Math.min(n, MAX_BUFFER_SIZE), buffer, 0);
      skip = buffer[0] - last - 1;
      bufferPosition = 0;
    }
  }

  /** Returns the used random generator. */
  public Random getRandomGenerator() {
    return this.sampler.getRandomGenerator();
  }

  /** Just shows how this class can be used; samples n elements from and int[] array. */
  public static int[] sampleArray(int n, int[] elements) {
    RandomSamplingAssistant assistant = new RandomSamplingAssistant(n, elements.length, null);
    int[] sample = new int[n];
    int j = 0;
    int length = elements.length;
    for (int i = 0; i < length; i++) {
      if (assistant.sampleNextElement()) {
        sample[j++] = elements[i];
      }
    }
    return sample;
  }

  /**
   * Returns whether the next element of the input sequence shall be sampled (picked) or not.
   *
   * @return <tt>true</tt> if the next element shall be sampled (picked), <tt>false</tt> otherwise.
   */
  public boolean sampleNextElement() {
    if (n == 0) {
      return false;
    } //reject
    if (skip-- > 0) {
      return false;
    } //reject

    //accept
    n--;
    if (bufferPosition < buffer.length - 1) {
      skip = buffer[bufferPosition + 1] - buffer[bufferPosition++];
      --skip;
    } else {
      fetchNextBlock();
    }

    return true;
  }

}
