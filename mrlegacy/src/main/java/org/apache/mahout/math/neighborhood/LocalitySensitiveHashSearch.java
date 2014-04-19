/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.neighborhood;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import org.apache.lucene.util.PriorityQueue;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.RandomProjector;
import org.apache.mahout.math.random.WeightedThing;
import org.apache.mahout.math.stats.OnlineSummarizer;

/**
 * Implements a Searcher that uses locality sensitivity hash as a first pass approximation
 * to estimate distance without floating point math.  The clever bit about this implementation
 * is that it does an adaptive cutoff for the cutoff on the bitwise distance.  Making this
 * cutoff adaptive means that we only needs to make a single pass through the data.
 */
public class LocalitySensitiveHashSearch extends UpdatableSearcher {
  /**
   * Number of bits in the locality sensitive hash. 64 bits fix neatly into a long.
   */
  private static final int BITS = 64;

  /**
   * Bit mask for the computed hash. Currently, it's 0xffffffffffff.
   */
  private static final long BIT_MASK = -1L;

  /**
   * The maximum Hamming distance between two hashes that the hash limit can grow back to.
   * It starts at BITS and decreases as more points than are needed are added to the candidate priority queue.
   * But, after the observed distribution of distances becomes too good (we're seeing less than some percentage of the
   * total number of points; using the hash strategy somewhere less than 25%) the limit is increased to compute
   * more distances.
   * This is because
   */
  private static final int MAX_HASH_LIMIT = 32;

  /**
   * Minimum number of points with a given Hamming from the query that must be observed to consider raising the minimum
   * distance for a candidate.
   */
  private static final int MIN_DISTRIBUTION_COUNT = 10;

  private final Multiset<HashedVector> trainingVectors = HashMultiset.create();

  /**
   * This matrix of BITS random vectors is used to compute the Locality Sensitive Hash
   * we compute the dot product with these vectors using a matrix multiplication and then use just
   * sign of each result as one bit in the hash
   */
  private Matrix projection;

  /**
   * The search size determines how many top results we retain.  We do this because the hash distance
   * isn't guaranteed to be entirely monotonic with respect to the real distance.  To the extent that
   * actual distance is well approximated by hash distance, then the searchSize can be decreased to
   * roughly the number of results that you want.
   */
  private int searchSize;

  /**
   * Controls how the hash limit is raised. 0 means use minimum of distribution, 1 means use first quartile.
   * Intermediate values indicate an interpolation should be used. Negative values mean to never increase.
   */
  private double hashLimitStrategy = 0.9;

  /**
   * Number of evaluations of the full distance between two points that was required.
   */
  private int distanceEvaluations = 0;

  /**
   * Whether the projection matrix was initialized. This has to be deferred until the size of the vectors is known,
   * effectively until the first vector is added.
   */
  private boolean initialized = false;

  public LocalitySensitiveHashSearch(DistanceMeasure distanceMeasure, int searchSize) {
    super(distanceMeasure);
    this.searchSize = searchSize;
    this.projection = null;
  }

  private void initialize(int numDimensions) {
    if (initialized) {
      return;
    }
    initialized = true;
    projection = RandomProjector.generateBasisNormal(BITS, numDimensions);
  }

  private PriorityQueue<WeightedThing<Vector>> searchInternal(Vector query) {
    long queryHash = HashedVector.computeHash64(query, projection);

    // We keep an approximation of the closest vectors here.
    PriorityQueue<WeightedThing<Vector>> top = Searcher.getCandidateQueue(getSearchSize());

    // We scan the vectors using bit counts as an approximation of the dot product so we can do as few
    // full distance computations as possible.  Our goal is to only do full distance computations for
    // vectors with hash distance at most as large as the searchSize biggest hash distance seen so far.

    OnlineSummarizer[] distribution = new OnlineSummarizer[BITS + 1];
    for (int i = 0; i < BITS + 1; i++) {
      distribution[i] = new OnlineSummarizer();
    }

    distanceEvaluations = 0;
    
    // We keep the counts of the hash distances here.  This lets us accurately
    // judge what hash distance cutoff we should use.
    int[] hashCounts = new int[BITS + 1];
    
    // Maximum number of different bits to still consider a vector a candidate for nearest neighbor.
    // Starts at the maximum number of bits, but decreases and can increase.
    int hashLimit = BITS;
    int limitCount = 0;
    double distanceLimit = Double.POSITIVE_INFINITY;

    // In this loop, we have the invariants that:
    //
    // limitCount = sum_{i<hashLimit} hashCount[i]
    // and
    // limitCount >= searchSize && limitCount - hashCount[hashLimit-1] < searchSize
    for (HashedVector vector : trainingVectors) {
      // This computes the Hamming Distance between the vector's hash and the query's hash.
      // The result is correlated with the angle between the vectors.
      int bitDot = vector.hammingDistance(queryHash);
      if (bitDot <= hashLimit) {
        distanceEvaluations++;

        double distance = distanceMeasure.distance(query, vector);
        distribution[bitDot].add(distance);

        if (distance < distanceLimit) {
          top.insertWithOverflow(new WeightedThing<Vector>(vector, distance));
          if (top.size() == searchSize) {
            distanceLimit = top.top().getWeight();
          }

          hashCounts[bitDot]++;
          limitCount++;
          while (hashLimit > 0 && limitCount - hashCounts[hashLimit - 1] > searchSize) {
            hashLimit--;
            limitCount -= hashCounts[hashLimit];
          }

          if (hashLimitStrategy >= 0) {
            while (hashLimit < MAX_HASH_LIMIT && distribution[hashLimit].getCount() > MIN_DISTRIBUTION_COUNT
                && ((1 - hashLimitStrategy) * distribution[hashLimit].getQuartile(0)
                + hashLimitStrategy * distribution[hashLimit].getQuartile(1)) < distanceLimit) {
              limitCount += hashCounts[hashLimit];
              hashLimit++;
            }
          }
        }
      }
    }
    return top;
  }

  @Override
  public List<WeightedThing<Vector>> search(Vector query, int limit) {
    PriorityQueue<WeightedThing<Vector>> top = searchInternal(query);
    List<WeightedThing<Vector>> results = Lists.newArrayListWithExpectedSize(top.size());
    while (top.size() != 0) {
      WeightedThing<Vector> wv = top.pop();
      results.add(new WeightedThing<Vector>(((HashedVector) wv.getValue()).getVector(), wv.getWeight()));
    }
    Collections.reverse(results);
    if (limit < results.size()) {
      results = results.subList(0, limit);
    }
    return results;
  }

  /**
   * Returns the closest vector to the query.
   * When only one the nearest vector is needed, use this method, NOT search(query, limit) because
   * it's faster (less overhead).
   * This is nearly the same as search().
   *
   * @param query the vector to search for
   * @param differentThanQuery if true, returns the closest vector different than the query (this
   *                           only matters if the query is among the searched vectors), otherwise,
   *                           returns the closest vector to the query (even the same vector).
   * @return the weighted vector closest to the query
   */
  @Override
  public WeightedThing<Vector> searchFirst(Vector query, boolean differentThanQuery) {
    // We get the top searchSize neighbors.
    PriorityQueue<WeightedThing<Vector>> top = searchInternal(query);
    // We then cut the number down to just the best 2.
    while (top.size() > 2) {
      top.pop();
    }
    // If there are fewer than 2 results, we just return the one we have.
    if (top.size() < 2) {
      return removeHash(top.pop());
    }
    // There are exactly 2 results.
    WeightedThing<Vector> secondBest = top.pop();
    WeightedThing<Vector> best = top.pop();
    // If the best result is the same as the query, but we don't want to return the query.
    if (differentThanQuery && best.getValue().equals(query)) {
      best = secondBest;
    }
    return removeHash(best);
  }

  protected static WeightedThing<Vector> removeHash(WeightedThing<Vector> input) {
    return new WeightedThing<Vector>(((HashedVector) input.getValue()).getVector(), input.getWeight());
  }

  @Override
  public void add(Vector vector) {
    initialize(vector.size());
    trainingVectors.add(new HashedVector(vector, projection, HashedVector.INVALID_INDEX, BIT_MASK));
  }

  @Override
  public int size() {
    return trainingVectors.size();
  }

  public int getSearchSize() {
    return searchSize;
  }

  public void setSearchSize(int size) {
    searchSize = size;
  }

  public void setRaiseHashLimitStrategy(double strategy) {
    hashLimitStrategy = strategy;
  }

  /**
   * This is only for testing.
   * @return the number of times the actual distance between two vectors was computed.
   */
  public int resetEvaluationCount() {
    int result = distanceEvaluations;
    distanceEvaluations = 0;
    return result;
  }

  @Override
  public Iterator<Vector> iterator() {
    return Iterators.transform(trainingVectors.iterator(), new Function<HashedVector, Vector>() {
      @Override
      public Vector apply(org.apache.mahout.math.neighborhood.HashedVector input) {
        Preconditions.checkNotNull(input);
        //noinspection ConstantConditions
        return input.getVector();
      }
    });
  }

  @Override
  public boolean remove(Vector v, double epsilon) {
    return trainingVectors.remove(new HashedVector(v, projection, HashedVector.INVALID_INDEX, BIT_MASK));
  }

  @Override
  public void clear() {
    trainingVectors.clear();
  }
}
