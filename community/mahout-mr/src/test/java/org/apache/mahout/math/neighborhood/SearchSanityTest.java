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


import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.jet.math.Constants;
import org.apache.mahout.math.random.MultiNormal;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

@RunWith(Parameterized.class)
public class SearchSanityTest extends MahoutTestCase {
  private static final int NUM_DATA_POINTS = 1 << 13;
  private static final int NUM_DIMENSIONS = 20;
  private static final int NUM_PROJECTIONS = 3;
  private static final int SEARCH_SIZE = 30;

  private UpdatableSearcher searcher;
  private Matrix dataPoints;

  protected static Matrix multiNormalRandomData(int numDataPoints, int numDimensions) {
    Matrix data = new DenseMatrix(numDataPoints, numDimensions);
    MultiNormal gen = new MultiNormal(20);
    for (MatrixSlice slice : data) {
      slice.vector().assign(gen.sample());
    }
    return data;
  }

  @Parameterized.Parameters
  public static List<Object[]> generateData() {
    RandomUtils.useTestSeed();
    Matrix dataPoints = multiNormalRandomData(NUM_DATA_POINTS, NUM_DIMENSIONS);
    return Arrays.asList(new Object[][]{
        {new ProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), dataPoints},
        {new FastProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE),
            dataPoints},
        {new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), SEARCH_SIZE), dataPoints},
    });
  }

  public SearchSanityTest(UpdatableSearcher searcher, Matrix dataPoints) {
    this.searcher = searcher;
    this.dataPoints = dataPoints;
  }

  @Test
  public void testExactMatch() {
    searcher.clear();
    Iterable<MatrixSlice> data = dataPoints;

    final Iterable<MatrixSlice> batch1 = Iterables.limit(data, 300);
    List<MatrixSlice> queries = Lists.newArrayList(Iterables.limit(batch1, 100));

    // adding the data in multiple batches triggers special code in some searchers
    searcher.addAllMatrixSlices(batch1);
    assertEquals(300, searcher.size());

    Vector q = Iterables.get(data, 0).vector();
    List<WeightedThing<Vector>> r = searcher.search(q, 2);
    assertEquals(0, r.get(0).getValue().minus(q).norm(1), 1.0e-8);

    final Iterable<MatrixSlice> batch2 = Iterables.limit(Iterables.skip(data, 300), 10);
    searcher.addAllMatrixSlices(batch2);
    assertEquals(310, searcher.size());

    q = Iterables.get(data, 302).vector();
    r = searcher.search(q, 2);
    assertEquals(0, r.get(0).getValue().minus(q).norm(1), 1.0e-8);

    searcher.addAllMatrixSlices(Iterables.skip(data, 310));
    assertEquals(dataPoints.numRows(), searcher.size());

    for (MatrixSlice query : queries) {
      r = searcher.search(query.vector(), 2);
      assertEquals("Distance has to be about zero", 0, r.get(0).getWeight(), 1.0e-6);
      assertEquals("Answer must be substantially the same as query", 0,
          r.get(0).getValue().minus(query.vector()).norm(1), 1.0e-8);
      assertTrue("Wrong answer must have non-zero distance",
          r.get(1).getWeight() > r.get(0).getWeight());
    }
  }

  @Test
  public void testNearMatch() {
    searcher.clear();
    List<MatrixSlice> queries = Lists.newArrayList(Iterables.limit(dataPoints, 100));
    searcher.addAllMatrixSlicesAsWeightedVectors(dataPoints);

    MultiNormal noise = new MultiNormal(0.01, new DenseVector(20));
    for (MatrixSlice slice : queries) {
      Vector query = slice.vector();
      final Vector epsilon = noise.sample();
      List<WeightedThing<Vector>> r = searcher.search(query, 2);
      query = query.plus(epsilon);
      assertEquals("Distance has to be small", epsilon.norm(2), r.get(0).getWeight(), 1.0e-1);
      assertEquals("Answer must be substantially the same as query", epsilon.norm(2),
          r.get(0).getValue().minus(query).norm(2), 1.0e-1);
      assertTrue("Wrong answer must be further away", r.get(1).getWeight() > r.get(0).getWeight());
    }
  }

  @Test
  public void testOrdering() {
    searcher.clear();
    Matrix queries = new DenseMatrix(100, 20);
    MultiNormal gen = new MultiNormal(20);
    for (int i = 0; i < 100; i++) {
      queries.viewRow(i).assign(gen.sample());
    }
    searcher.addAllMatrixSlices(dataPoints);

    for (MatrixSlice query : queries) {
      List<WeightedThing<Vector>> r = searcher.search(query.vector(), 200);
      double x = 0;
      for (WeightedThing<Vector> thing : r) {
        assertTrue("Scores must be monotonic increasing", thing.getWeight() >= x);
        x = thing.getWeight();
      }
    }
  }

  @Test
  public void testRemoval() {
    searcher.clear();
    searcher.addAllMatrixSlices(dataPoints);
    //noinspection ConstantConditions
    if (searcher instanceof UpdatableSearcher) {
      List<Vector> x = Lists.newArrayList(Iterables.limit(searcher, 2));
      int size0 = searcher.size();

      List<WeightedThing<Vector>> r0 = searcher.search(x.get(0), 2);

      searcher.remove(x.get(0), 1.0e-7);
      assertEquals(size0 - 1, searcher.size());

      List<WeightedThing<Vector>> r = searcher.search(x.get(0), 1);
      assertTrue("Vector should be gone", r.get(0).getWeight() > 0);
      assertEquals("Previous second neighbor should be first", 0,
          r.get(0).getValue().minus(r0.get(1).getValue()).norm (1), 1.0e-8);

      searcher.remove(x.get(1), 1.0e-7);
      assertEquals(size0 - 2, searcher.size());

      r = searcher.search(x.get(1), 1);
      assertTrue("Vector should be gone", r.get(0).getWeight() > 0);

      // Vectors don't show up in iterator.
      for (Vector v : searcher) {
        assertTrue(x.get(0).minus(v).norm(1) > 1.0e-6);
        assertTrue(x.get(1).minus(v).norm(1) > 1.0e-6);
      }
    } else {
      try {
        List<Vector> x = Lists.newArrayList(Iterables.limit(searcher, 2));
        searcher.remove(x.get(0), 1.0e-7);
        fail("Shouldn't be able to delete from " + searcher.getClass().getName());
      } catch (UnsupportedOperationException e) {
        // good enough that UOE is thrown
      }
    }
  }

  @Test
  public void testSearchFirst() {
    searcher.clear();
    searcher.addAll(dataPoints);
    for (Vector datapoint : dataPoints) {
      WeightedThing<Vector> first = searcher.searchFirst(datapoint, false);
      WeightedThing<Vector> second = searcher.searchFirst(datapoint, true);
      List<WeightedThing<Vector>> firstTwo = searcher.search(datapoint, 2);

      assertEquals("First isn't self", 0, first.getWeight(), 0);
      assertEquals("First isn't self", datapoint, first.getValue());
      assertEquals("First doesn't match", first, firstTwo.get(0));
      assertEquals("Second doesn't match", second, firstTwo.get(1));
    }
  }

  @Test
  public void testSearchLimiting() {
    searcher.clear();
    searcher.addAll(dataPoints);
    for (Vector datapoint : dataPoints) {
      List<WeightedThing<Vector>> firstTwo = searcher.search(datapoint, 2);

      assertThat("Search limit isn't respected", firstTwo.size(), is(lessThanOrEqualTo(2)));
    }
  }

  @Test
  public void testRemove() {
    searcher.clear();
    for (int i = 0; i < dataPoints.rowSize(); ++i) {
      Vector datapoint = dataPoints.viewRow(i);
      searcher.add(datapoint);
      // As long as points are not searched for right after being added, in FastProjectionSearch, points are not
      // merged with the main list right away, so if a search for a point occurs before it's merged the pendingAdditions
      // list also needs to be looked at.
      // This used to not be the case for searchFirst(), thereby causing removal failures.
      if (i % 2 == 0) {
        assertTrue("Failed to find self [search]",
            searcher.search(datapoint, 1).get(0).getWeight() < Constants.EPSILON);
        assertTrue("Failed to find self [searchFirst]",
            searcher.searchFirst(datapoint, false).getWeight() < Constants.EPSILON);
        assertTrue("Failed to remove self", searcher.remove(datapoint, Constants.EPSILON));
      }
    }
  }
}
