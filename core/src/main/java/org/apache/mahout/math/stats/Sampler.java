package org.apache.mahout.math.stats;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.Random;

/**
 * Discrete distribution sampler:
 *
 * Samples from a given discrete distribution: you provide a source of randomness and a Vector
 * (cardinality N) which describes a distribution over [0,N), and calls to sample() sample
 * from 0 to N using this distribution
 *
 */
public class Sampler {

  private Random random;
  private double[] sampler;

  public Sampler(Random random) {
    this.random = random;
    sampler = null;
  }

  public Sampler(Random random, double[] sampler) {
    this.random = random;
    this.sampler = sampler;
  }

  public Sampler(Random random, Vector distribution) {
    this.random = random;
    this.sampler = samplerFor(distribution);
  }

  public int sample(Vector distribution) {
    return sample(samplerFor(distribution));
  }

  public int sample() {
    if(sampler == null) {
      throw new NullPointerException("Sampler must have been constructed with a distribution, or"
        + " else sample(Vector) should be used to sample");
    }
    return sample(sampler);
  }

  private double[] samplerFor(double[] distribution) {
    return samplerFor(new DenseVector(distribution));
  }

  private double[] samplerFor(Vector vectorDistribution) {
    int size = vectorDistribution.size();
    double[] partition = new double[size];
    double norm = vectorDistribution.norm(1);
    double sum = 0;
    for(int i = 0; i < size; i++) {
      sum += (vectorDistribution.get(i) / norm);
      partition[i] = sum;
    }
    return partition;
  }

  private int sample(double[] sampler) {
    int index = Arrays.binarySearch(sampler, random.nextDouble());
    return index < 0 ? -(index+1) : index;
  }
}
