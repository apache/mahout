package org.apache.mahout.math.neighborhood;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.math.random.RandomProjector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.WeightedThing;

/**
 * Does approximate nearest neighbor search by projecting the vectors similar to ProjectionSearch.
 * The main difference between this class and the ProjectionSearch is the use of sorted arrays
 * instead of binary search trees to implement the sets of scalar projections.
 *
 * Instead of taking log n time to add a vector to each of the vectors, * the pending additions are
 * kept separate and are searched using a brute search. When there are "enough" pending additions,
 * they're committed into the main pool of vectors.
 */
public class FastProjectionSearch extends UpdatableSearcher {
  // The list of vectors that have not yet been projected (that are pending).
  private List<Vector> pendingAdditions = Lists.newArrayList();

  // The list of basis vectors. Populated when the first vector's dimension is know by calling
  // initialize once.
  private Matrix basisMatrix = null;

  // The list of sorted lists of scalar projections. The outer list has one entry for each basis
  // vector that all the other vectors will be projected on.
  // For each basis vector, the inner list has an entry for each vector that has been projected.
  // These entries are WeightedThing<Vector> where the weight is the value of the scalar
  // projection and the value is the vector begin referred to.
  private List<List<WeightedThing<Vector>>> scalarProjections;

  // The number of projection used for approximating the distance.
  private int numProjections;

  // The number of elements to keep on both sides of the closest estimated distance as possible
  // candidates for the best actual distance.
  private int searchSize;

  // Initially, the dimension of the vectors searched by this searcher is unknown. After adding
  // the first vector, the basis will be initialized. This marks whether initialization has
  // happened or not so we only do it once.
  private boolean initialized = false;

  // Whether the iterator returned from the searcher was used to modify any of the vectors. This
  // flag must be set manually by calling setDirty after said modification so the internal
  // structures can be updated.
  private boolean dirty = false;

  // Removing vectors from the searcher is done lazily to avoid the linear time cost of removing
  // elements from an array. This member keeps track of the number of removed vectors (marked as
  // "impossible" values in the array) so they can be removed when updating the structure.
  private int numPendingRemovals = 0;

  private final static double ADDITION_THRESHOLD = 0.05;
  private final static double REMOVAL_THRESHOLD = 0.02;

  public FastProjectionSearch(DistanceMeasure distanceMeasure, int numProjections, int searchSize) {
    super(distanceMeasure);
    Preconditions.checkArgument(numProjections > 0 && numProjections < 100,
        "Unreasonable value for number of projections");
    this.numProjections = numProjections;
    this.searchSize = searchSize;
    scalarProjections = Lists.newArrayListWithCapacity(numProjections);
    for (int i = 0; i < numProjections; ++i) {
      scalarProjections.add(Lists.<WeightedThing<Vector>>newArrayList());
    }
  }

  private void initialize(int numDimensions) {
    if (initialized) {
      return;
    }
    basisMatrix = RandomProjector.generateBasisNormal(numProjections, numDimensions);
    initialized = true;
  }

  /**
   * Add a new Vector to the Searcher that will be checked when getting
   * the nearest neighbors.
   * <p/>
   * The vector IS NOT CLONED. Do not modify the vector externally otherwise the internal
   * Searcher data structures could be invalidated.
   */
  @Override
  public void add(Vector vector) {
    initialize(vector.size());
    pendingAdditions.add(vector);
  }

  /**
   * Returns the number of WeightedVectors being searched for nearest neighbors.
   */
  @Override
  public int size() {
    return pendingAdditions.size() + scalarProjections.get(0).size() - numPendingRemovals;
  }

  /**
   * When querying the Searcher for the closest vectors, a list of WeightedThing<Vector>s is
   * returned. The value of the WeightedThing is the neighbor and the weight is the
   * the distance (calculated by some metric - see a concrete implementation) between the query
   * and neighbor.
   * The actual type of vector in the pair is the same as the vector added to the Searcher.
   */
  @Override
  public List<WeightedThing<Vector>> search(Vector query, int limit) {
    reindex(false);

    Set<Vector> candidates = Sets.newHashSet();
    Vector projection = basisMatrix.times(query);
    for (int i = 0; i < basisMatrix.numRows(); ++i) {
      List<WeightedThing<Vector>> currProjections = scalarProjections.get(i);
      int middle = Collections.binarySearch(currProjections,
          new WeightedThing<Vector>(projection.get(i)));
      if (middle < 0) {
        middle = -(middle + 1);
      }
      for (int j = Math.max(0, middle - searchSize);
           j < Math.min(currProjections.size(), middle + searchSize + 1); ++j) {
        if (currProjections.get(j).getValue() == null) {
          continue;
        }
        candidates.add(currProjections.get(j).getValue());
      }
    }

    List<WeightedThing<Vector>> top =
        Lists.newArrayListWithCapacity(candidates.size() + pendingAdditions.size());
    for (Vector candidate : Iterables.concat(candidates, pendingAdditions)) {
      top.add(new WeightedThing<Vector>(candidate, distanceMeasure.distance(candidate, query)));
    }
    Collections.sort(top);

    return top.subList(0, Math.min(top.size(), limit));
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
    reindex(false);

    double bestDistance = Double.POSITIVE_INFINITY;
    Vector bestVector = null;

    Vector projection = basisMatrix.times(query);
    for (int i = 0; i < basisMatrix.numRows(); ++i) {
      List<WeightedThing<Vector>> currProjections = scalarProjections.get(i);
      int middle = Collections.binarySearch(currProjections,
          new WeightedThing<Vector>(projection.get(i)));
      if (middle < 0) {
        middle = -(middle + 1);
      }
      for (int j = Math.max(0, middle - searchSize);
           j < Math.min(currProjections.size(), middle + searchSize + 1); ++j) {
        if (currProjections.get(j).getValue() == null) {
          continue;
        }
        Vector vector = currProjections.get(j).getValue();
        double distance = distanceMeasure.distance(vector, query);
        if (distance < bestDistance && (!differentThanQuery || !vector.equals(query))) {
          bestDistance = distance;
          bestVector = vector;
        }
      }
    }

    return new WeightedThing<Vector>(bestVector, bestDistance);
  }

  @Override
  public boolean remove(Vector v, double epsilon) {
    WeightedThing<Vector> closestPair = searchFirst(v, false);
    if (distanceMeasure.distance(closestPair.getValue(), v) > epsilon) {
      return false;
    }

    boolean isProjected = true;
    final Vector projection = basisMatrix.times(v);
    for (int i = 0; i < basisMatrix.numRows(); ++i) {
      List<WeightedThing<Vector>> currProjections = scalarProjections.get(i);
      int middle = Collections.binarySearch(currProjections,
          new WeightedThing<Vector>(projection.get(i)));
      if (middle < 0) {
        isProjected = false;
        break;
      }
      double oldWeight = currProjections.get(middle).getWeight();
      scalarProjections.get(i).set(middle, new WeightedThing<Vector>(oldWeight));
    }
    if (isProjected) {
      ++numPendingRemovals;
      return true;
    }

    for (int i = 0; i < pendingAdditions.size(); ++i) {
      if (distanceMeasure.distance(v, pendingAdditions.get(i)) < epsilon) {
        pendingAdditions.remove(i);
        break;
      }
    }
    return true;
  }

  private void reindex(boolean force) {
    int numProjected = scalarProjections.get(0).size();
    if (force || dirty || pendingAdditions.size() > ADDITION_THRESHOLD * numProjected ||
        numPendingRemovals > REMOVAL_THRESHOLD * numProjected) {

      // We only need to copy the first list because when iterating we use only that list for the Vector
      // references.
      // see public Iterator<Vector> iterator()
      List<List<WeightedThing<Vector>>> scalarProjections = Lists.newArrayListWithCapacity(numProjections);
      for (int i = 0; i < numProjections; ++i) {
        if (i == 0) {
          scalarProjections.add(Lists.newArrayList(this.scalarProjections.get(i)));
        } else {
          scalarProjections.add(this.scalarProjections.get(i));
        }
      }

      // Project every pending vector onto every basis vector.
      for (Vector pending : pendingAdditions) {
        Vector projection = basisMatrix.times(pending);
        for (int i = 0; i < numProjections; ++i) {
          scalarProjections.get(i).add(new WeightedThing<Vector>(pending, projection.get(i)));
        }
      }
      pendingAdditions.clear();
      // For each basis vector, sort the resulting list (for binary search) and remove the number
      // of pending removals (it's the same for every basis vector) at the end (the weights are
      // set to Double.POSITIVE_INFINITY when removing).
      for (int i = 0; i < numProjections; ++i) {
        List<WeightedThing<Vector>> currProjections = scalarProjections.get(i);
        for (WeightedThing<Vector> v : currProjections) {
          if (v.getValue() == null) {
            v.setWeight(Double.POSITIVE_INFINITY);
          }
        }
        Collections.sort(currProjections);
        for (int j = 0; j < numPendingRemovals; ++j) {
          currProjections.remove(currProjections.size() - 1);
        }
      }
      numPendingRemovals = 0;

      this.scalarProjections = scalarProjections;
    }
  }

  @Override
  public void clear() {
    pendingAdditions.clear();
    for (int i = 0; i < numProjections; ++i) {
      scalarProjections.get(i).clear();
    }
    numPendingRemovals = 0;
    dirty = false;
  }

  /**
   * This iterates on the snapshot of the contents first instantiated regardless of any future modifications.
   * Changes done after the iterator is created will not be visible to the iterator but will be visible
   * when searching.
   * @return iterator through the vectors in this searcher.
   */
  @Override
  public Iterator<Vector> iterator() {
    reindex(true);
    return new AbstractIterator<Vector>() {
          Iterator<WeightedThing<Vector>> data = scalarProjections.get(0).iterator();
          @Override
          protected Vector computeNext() {
            WeightedThing<Vector> next;
            do {
              if (!data.hasNext()) {
                return endOfData();
              }
              next = data.next();
              if (next.getValue() != null) {
                return next.getValue();
              }
            } while (true);
          }
        };
  }

  /**
   * When modifying an element of the searcher through the iterator,
   * the user MUST CALL setDirty() to update the internal data structures. Otherwise,
   * the internal order of the vectors will change and future results might be wrong.
   */
  @SuppressWarnings("unused")
  public void setDirty() {
    dirty = true;
  }
}
