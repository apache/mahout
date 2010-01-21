/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random.sampling;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.jet.random.Uniform;
import org.apache.mahout.math.jet.random.engine.RandomEngine;
import org.apache.mahout.math.list.IntArrayList;

/**
 * Conveniently computes a stable subsequence of elements from a given input sequence;
 * Picks (samples) exactly one random element from successive blocks of <tt>weight</tt> input elements each.
 * For example, if weight==2 (a block is 2 elements), and the input is 5*2=10 elements long, then picks 5 random elements from the 10 elements such that
 * one element is randomly picked from the first block, one element from the second block, ..., one element from the last block.
 * weight == 1.0 --> all elements are picked (sampled). weight == 10.0 --> Picks one random element from successive blocks of 10 elements each. Etc.
 * The subsequence is guaranteed to be <i>stable</i>, i.e. elements never change position relative to each other.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class WeightedRandomSampler extends PersistentObject {

  //public class BlockedRandomSampler extends Object implements java.io.Serializable {
  private int skip;
  private int nextTriggerPos;
  private int nextSkip;
  private int weight;
  private Uniform generator;

  private static final int UNDEFINED = -1;

  /** Calls <tt>BlockedRandomSampler(1,null)</tt>. */
  public WeightedRandomSampler() {
    this(1, null);
  }

  /**
   * Chooses exactly one random element from successive blocks of <tt>weight</tt> input elements each. For example, if
   * weight==2, and the input is 5*2=10 elements long, then chooses 5 random elements from the 10 elements such that one
   * is chosen from the first block, one from the second, ..., one from the last block. weight == 1.0 --> all elements
   * are consumed (sampled). 10.0 --> Consumes one random element from successive blocks of 10 elements each. Etc.
   *
   * @param weight          the weight.
   * @param randomGenerator a random number generator. Set this parameter to <tt>null</tt> to use the default random
   *                        number generator.
   */
  public WeightedRandomSampler(int weight, RandomEngine randomGenerator) {
    if (randomGenerator == null) {
      randomGenerator = org.apache.mahout.math.jet.random.AbstractDistribution.makeDefaultGenerator();
    }
    this.generator = new Uniform(randomGenerator);
    setWeight(weight);
  }

  /** Returns a deep copy of the receiver. */
  @Override
  public Object clone() {
    WeightedRandomSampler copy = (WeightedRandomSampler) super.clone();
    copy.generator = (Uniform) this.generator.clone();
    return copy;
  }

  public int getWeight() {
    return this.weight;
  }

  /**
   * Chooses exactly one random element from successive blocks of <tt>weight</tt> input elements each. For example, if
   * weight==2, and the input is 5*2=10 elements long, then chooses 5 random elements from the 10 elements such that one
   * is chosen from the first block, one from the second, ..., one from the last block.
   *
   * @return <tt>true</tt> if the next element shall be sampled (picked), <tt>false</tt> otherwise.
   */
  public boolean sampleNextElement() {
    if (skip > 0) { //reject
      skip--;
      return false;
    }

    if (nextTriggerPos == UNDEFINED) {
      if (weight == 1) {
        nextTriggerPos = 0; // tuned for speed
      } else {
        nextTriggerPos = generator.nextIntFromTo(0, weight - 1);
      }

      nextSkip = weight - 1 - nextTriggerPos;
    }

    if (nextTriggerPos > 0) { //reject
      nextTriggerPos--;
      return false;
    }

    //accept
    nextTriggerPos = UNDEFINED;
    skip = nextSkip;

    return true;
  }

  /**
   * Not yet commented.
   *
   * @param weight int
   */
  public void setWeight(int weight) {
    if (weight < 1) {
      throw new IllegalArgumentException("bad weight");
    }
    this.weight = weight;
    this.skip = 0;
    this.nextTriggerPos = UNDEFINED;
    this.nextSkip = 0;
  }

  /** Not yet commented. */
  public static void test(int weight, int size) {
    WeightedRandomSampler sampler = new WeightedRandomSampler();
    sampler.setWeight(weight);

    IntArrayList sample = new IntArrayList();
    for (int i = 0; i < size; i++) {
      if (sampler.sampleNextElement()) {
        sample.add(i);
      }
    }
  }

}
