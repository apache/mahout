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

import java.util.List;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.lucene.util.PriorityQueue;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.random.WeightedThing;

/**
 * Describes how to search a bunch of vectors.
 * The vectors can be of any type (weighted, sparse, ...) but only the values of the vector  matter
 * when searching (weights, indices, ...) will not.
 *
 * When iterating through a Searcher, the Vectors added to it are returned.
 */
public abstract class Searcher implements Iterable<Vector> {
  protected DistanceMeasure distanceMeasure;

  protected Searcher(DistanceMeasure distanceMeasure) {
    this.distanceMeasure = distanceMeasure;
  }

  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  /**
   * Add a new Vector to the Searcher that will be checked when getting
   * the nearest neighbors.
   *
   * The vector IS NOT CLONED. Do not modify the vector externally otherwise the internal
   * Searcher data structures could be invalidated.
   */
  public abstract void add(Vector vector);

  /**
   * Returns the number of WeightedVectors being searched for nearest neighbors.
   */
  public abstract int size();

  /**
   * When querying the Searcher for the closest vectors, a list of WeightedThing<Vector>s is
   * returned. The value of the WeightedThing is the neighbor and the weight is the
   * the distance (calculated by some metric - see a concrete implementation) between the query
   * and neighbor.
   * The actual type of vector in the pair is the same as the vector added to the Searcher.
   * @param query the vector to search for
   * @param limit the number of results to return
   * @return the list of weighted vectors closest to the query
   */
  public abstract List<WeightedThing<Vector>> search(Vector query, int limit);

  public List<List<WeightedThing<Vector>>> search(Iterable<? extends Vector> queries, int limit) {
    List<List<WeightedThing<Vector>>> results = Lists.newArrayListWithExpectedSize(Iterables.size(queries));
    for (Vector query : queries) {
      results.add(search(query, limit));
    }
    return results;
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
  public abstract WeightedThing<Vector> searchFirst(Vector query, boolean differentThanQuery);

  public List<WeightedThing<Vector>> searchFirst(Iterable<? extends Vector> queries, boolean differentThanQuery) {
    List<WeightedThing<Vector>> results = Lists.newArrayListWithExpectedSize(Iterables.size(queries));
    for (Vector query : queries) {
      results.add(searchFirst(query, differentThanQuery));
    }
    return results;
  }

  /**
   * Adds all the data elements in the Searcher.
   *
   * @param data an iterable of WeightedVectors to add.
   */
  public void addAll(Iterable<? extends Vector> data) {
    for (Vector vector : data) {
      add(vector);
    }
  }

  /**
   * Adds all the data elements in the Searcher.
   *
   * @param data an iterable of MatrixSlices to add.
   */
  public void addAllMatrixSlices(Iterable<MatrixSlice> data) {
    for (MatrixSlice slice : data) {
      add(slice.vector());
    }
  }

  public void addAllMatrixSlicesAsWeightedVectors(Iterable<MatrixSlice> data) {
    for (MatrixSlice slice : data) {
      add(new WeightedVector(slice.vector(), 1, slice.index()));
    }
  }

  public boolean remove(Vector v, double epsilon) {
    throw new UnsupportedOperationException("Can't remove a vector from a "
        + this.getClass().getName());
  }

  public void clear() {
    throw new UnsupportedOperationException("Can't remove vectors from a "
        + this.getClass().getName());
  }

  /**
   * Returns a bounded size priority queue, in reverse order that keeps track of the best nearest neighbor vectors.
   * @param limit maximum size of the heap.
   * @return the priority queue.
   */
  public static PriorityQueue<WeightedThing<Vector>> getCandidateQueue(int limit) {
    return new PriorityQueue<WeightedThing<Vector>>(limit) {
      @Override
      protected boolean lessThan(WeightedThing<Vector> a, WeightedThing<Vector> b) {
        return a.getWeight() > b.getWeight();
      }
    };
  }
}
