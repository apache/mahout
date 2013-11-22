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
import java.util.Set;

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.BoundType;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.collect.TreeMultiset;
import org.apache.mahout.math.random.RandomProjector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.WeightedThing;

/**
 * Does approximate nearest neighbor dudes search by projecting the data.
 */
public class ProjectionSearch extends UpdatableSearcher {

  /**
   * A lists of tree sets containing the scalar projections of each vector.
   * The elements in a TreeMultiset are WeightedThing<Integer>, where the weight is the scalar
   * projection of the vector at the index pointed to by the Integer from the referenceVectors list
   * on the basis vector whose index is the same as the index of the TreeSet in the List.
   */
  private List<TreeMultiset<WeightedThing<Vector>>> scalarProjections;

  /**
   * The list of random normalized projection vectors forming a basis.
   * The TreeSet of scalar projections at index i in scalarProjections corresponds to the vector
   * at index i from basisVectors.
   */
  private Matrix basisMatrix;

  /**
   * The number of elements to consider on both sides in the ball around the vector found by the
   * search in a TreeSet from scalarProjections.
   */
  private final int searchSize;

  private final int numProjections;
  private boolean initialized = false;

  private void initialize(int numDimensions) {
    if (initialized) {
      return;
    }
    initialized = true;
    basisMatrix = RandomProjector.generateBasisNormal(numProjections, numDimensions);
    scalarProjections = Lists.newArrayList();
    for (int i = 0; i < numProjections; ++i) {
      scalarProjections.add(TreeMultiset.<WeightedThing<Vector>>create());
    }
  }

  public ProjectionSearch(DistanceMeasure distanceMeasure, int numProjections,  int searchSize) {
    super(distanceMeasure);
    Preconditions.checkArgument(numProjections > 0 && numProjections < 100,
        "Unreasonable value for number of projections. Must be: 0 < numProjections < 100");

    this.searchSize = searchSize;
    this.numProjections = numProjections;
  }

  /**
   * Adds a WeightedVector into the set of projections for later searching.
   * @param vector  The WeightedVector to add.
   */
  @Override
  public void add(Vector vector) {
    initialize(vector.size());
    Vector projection = basisMatrix.times(vector);
    // Add the the new vector and the projected distance to each set separately.
    int i = 0;
    for (TreeMultiset<WeightedThing<Vector>> s : scalarProjections) {
      s.add(new WeightedThing<Vector>(vector, projection.get(i++)));
    }
    int numVectors = scalarProjections.get(0).size();
    for (TreeMultiset<WeightedThing<Vector>> s : scalarProjections) {
      Preconditions.checkArgument(s.size() == numVectors, "Number of vectors in projection sets "
          + "differ");
      double firstWeight = s.firstEntry().getElement().getWeight();
      for (WeightedThing<Vector> w : s) {
        Preconditions.checkArgument(firstWeight <= w.getWeight(), "Weights not in non-decreasing "
            + "order");
        firstWeight = w.getWeight();
      }
    }
  }

  /**
   * Returns the number of scalarProjections that we can search
   * @return  The number of scalarProjections added to the search so far.
   */
  @Override
  public int size() {
    if (scalarProjections == null) {
      return 0;
    }
    return scalarProjections.get(0).size();
  }

  /**
   * Searches for the query vector returning the closest limit referenceVectors.
   *
   * @param query the vector to search for.
   * @param limit the number of results to return.
   * @return a list of Vectors wrapped in WeightedThings where the "thing"'s weight is the
   * distance.
   */
  @Override
  public List<WeightedThing<Vector>> search(Vector query, int limit) {
    Set<Vector> candidates = Sets.newHashSet();

    Iterator<? extends Vector> projections = basisMatrix.iterator();
    for (TreeMultiset<WeightedThing<Vector>> v : scalarProjections) {
      Vector basisVector = projections.next();
      WeightedThing<Vector> projectedQuery = new WeightedThing<Vector>(query,
          query.dot(basisVector));
      for (WeightedThing<Vector> candidate : Iterables.concat(
          Iterables.limit(v.tailMultiset(projectedQuery, BoundType.CLOSED), searchSize),
          Iterables.limit(v.headMultiset(projectedQuery, BoundType.OPEN).descendingMultiset(), searchSize))) {
        candidates.add(candidate.getValue());
      }
    }

    // If searchSize * scalarProjections.size() is small enough not to cause much memory pressure,
    // this is probably just as fast as a priority queue here.
    List<WeightedThing<Vector>> top = Lists.newArrayList();
    for (Vector candidate : candidates) {
      top.add(new WeightedThing<Vector>(candidate, distanceMeasure.distance(query, candidate)));
    }
    Collections.sort(top);
    return top.subList(0, Math.min(limit, top.size()));
  }

  /**
   * Returns the closest vector to the query.
   * When only one the nearest vector is needed, use this method, NOT search(query, limit) because
   * it's faster (less overhead).
   *
   * @param query the vector to search for
   * @param differentThanQuery if true, returns the closest vector different than the query (this
   *                           only matters if the query is among the searched vectors), otherwise,
   *                           returns the closest vector to the query (even the same vector).
   * @return the weighted vector closest to the query
   */
  @Override
  public WeightedThing<Vector> searchFirst(Vector query, boolean differentThanQuery) {
    double bestDistance = Double.POSITIVE_INFINITY;
    Vector bestVector = null;

    Iterator<? extends Vector> projections = basisMatrix.iterator();
    for (TreeMultiset<WeightedThing<Vector>> v : scalarProjections) {
      Vector basisVector = projections.next();
      WeightedThing<Vector> projectedQuery = new WeightedThing<Vector>(query, query.dot(basisVector));
      for (WeightedThing<Vector> candidate : Iterables.concat(
          Iterables.limit(v.tailMultiset(projectedQuery, BoundType.CLOSED), searchSize),
          Iterables.limit(v.headMultiset(projectedQuery, BoundType.OPEN).descendingMultiset(), searchSize))) {
        double distance = distanceMeasure.distance(query, candidate.getValue());
        if (distance < bestDistance && (!differentThanQuery || !candidate.getValue().equals(query))) {
          bestDistance = distance;
          bestVector = candidate.getValue();
        }
      }
    }

    return new WeightedThing<Vector>(bestVector, bestDistance);
  }

  @Override
  public Iterator<Vector> iterator() {
    return new AbstractIterator<Vector>() {
      private final Iterator<WeightedThing<Vector>> projected = scalarProjections.get(0).iterator();
      @Override
      protected Vector computeNext() {
        if (!projected.hasNext()) {
          return endOfData();
        }
        return projected.next().getValue();
      }
    };
  }

  @Override
  public boolean remove(Vector vector, double epsilon) {
    WeightedThing<Vector> toRemove = searchFirst(vector, false);
    if (toRemove.getWeight() < epsilon) {
      Iterator<? extends Vector> basisVectors = basisMatrix.iterator();
      for (TreeMultiset<WeightedThing<Vector>> projection : scalarProjections) {
        if (!projection.remove(new WeightedThing<Vector>(vector, vector.dot(basisVectors.next())))) {
          throw new RuntimeException("Internal inconsistency in ProjectionSearch");
        }
      }
      return true;
    } else {
      return false;
    }
  }

  @Override
  public void clear() {
    if (scalarProjections == null) {
      return;
    }
    for (TreeMultiset<WeightedThing<Vector>> set : scalarProjections) {
      set.clear();
    }
  }
}
