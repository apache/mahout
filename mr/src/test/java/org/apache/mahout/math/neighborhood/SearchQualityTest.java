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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(Parameterized.class)
public class SearchQualityTest {
  private static final int NUM_DATA_POINTS = 1 << 14;
  private static final int NUM_QUERIES = 1 << 10;
  private static final int NUM_DIMENSIONS = 40;
  private static final int NUM_RESULTS = 2;

  private final Searcher searcher;
  private final Matrix dataPoints;
  private final Matrix queries;
  private Pair<List<List<WeightedThing<Vector>>>, Long> reference;
  private Pair<List<WeightedThing<Vector>>, Long> referenceSearchFirst;

  @Parameterized.Parameters
  public static List<Object[]> generateData() {
    RandomUtils.useTestSeed();
    Matrix dataPoints = LumpyData.lumpyRandomData(NUM_DATA_POINTS, NUM_DIMENSIONS);
    Matrix queries = LumpyData.lumpyRandomData(NUM_QUERIES, NUM_DIMENSIONS);

    DistanceMeasure distanceMeasure = new CosineDistanceMeasure();

    Searcher bruteSearcher = new BruteSearch(distanceMeasure);
    bruteSearcher.addAll(dataPoints);
    Pair<List<List<WeightedThing<Vector>>>, Long> reference = getResultsAndRuntime(bruteSearcher, queries);

    Pair<List<WeightedThing<Vector>>, Long> referenceSearchFirst =
        getResultsAndRuntimeSearchFirst(bruteSearcher, queries);

    double bruteSearchAvgTime = reference.getSecond() / (queries.numRows() * 1.0);
    System.out.printf("BruteSearch: avg_time(1 query) %f[s]\n", bruteSearchAvgTime);

    return Arrays.asList(new Object[][]{
        // NUM_PROJECTIONS = 3
        // SEARCH_SIZE = 10
        {new ProjectionSearch(distanceMeasure, 3, 10), dataPoints, queries, reference, referenceSearchFirst},
        {new FastProjectionSearch(distanceMeasure, 3, 10), dataPoints, queries, reference, referenceSearchFirst},
        // NUM_PROJECTIONS = 5
        // SEARCH_SIZE = 5
        {new ProjectionSearch(distanceMeasure, 5, 5), dataPoints, queries, reference, referenceSearchFirst},
        {new FastProjectionSearch(distanceMeasure, 5, 5), dataPoints, queries, reference, referenceSearchFirst},
    }
    );
  }

  public SearchQualityTest(Searcher searcher, Matrix dataPoints, Matrix queries,
                           Pair<List<List<WeightedThing<Vector>>>, Long> reference,
                           Pair<List<WeightedThing<Vector>>, Long> referenceSearchFirst) {
    this.searcher = searcher;
    this.dataPoints = dataPoints;
    this.queries = queries;
    this.reference = reference;
    this.referenceSearchFirst = referenceSearchFirst;
  }

  @Test
  public void testOverlapAndRuntimeSearchFirst() {
    searcher.clear();
    searcher.addAll(dataPoints);
    Pair<List<WeightedThing<Vector>>, Long> results = getResultsAndRuntimeSearchFirst(searcher, queries);

    int numFirstMatches = 0;
    for (int i = 0; i < queries.numRows(); ++i) {
      WeightedThing<Vector> referenceVector = referenceSearchFirst.getFirst().get(i);
      WeightedThing<Vector> resultVector = results.getFirst().get(i);
      if (referenceVector.getValue().equals(resultVector.getValue())) {
        ++numFirstMatches;
      }
    }

    double bruteSearchAvgTime = reference.getSecond() / (queries.numRows() * 1.0);
    double searcherAvgTime = results.getSecond() / (queries.numRows() * 1.0);
    System.out.printf("%s: first matches %d [%d]; avg_time(1 query) %f(s) [%f]\n",
        searcher.getClass().getName(), numFirstMatches, queries.numRows(),
        searcherAvgTime, bruteSearchAvgTime);

    assertEquals("Closest vector returned doesn't match", queries.numRows(), numFirstMatches);
    assertTrue("Searcher " + searcher.getClass().getName() + " slower than brute",
        bruteSearchAvgTime > searcherAvgTime);
  }
  @Test
  public void testOverlapAndRuntime() {
    searcher.clear();
    searcher.addAll(dataPoints);
    Pair<List<List<WeightedThing<Vector>>>, Long> results = getResultsAndRuntime(searcher, queries);

    int numFirstMatches = 0;
    int numMatches = 0;
    StripWeight stripWeight = new StripWeight();
    for (int i = 0; i < queries.numRows(); ++i) {
      List<WeightedThing<Vector>> referenceVectors = reference.getFirst().get(i);
      List<WeightedThing<Vector>> resultVectors = results.getFirst().get(i);
      if (referenceVectors.get(0).getValue().equals(resultVectors.get(0).getValue())) {
        ++numFirstMatches;
      }
      for (Vector v : Iterables.transform(referenceVectors, stripWeight)) {
        for (Vector w : Iterables.transform(resultVectors, stripWeight)) {
          if (v.equals(w)) {
            ++numMatches;
          }
        }
      }
    }

    double bruteSearchAvgTime = reference.getSecond() / (queries.numRows() * 1.0);
    double searcherAvgTime = results.getSecond() / (queries.numRows() * 1.0);
    System.out.printf("%s: first matches %d [%d]; total matches %d [%d]; avg_time(1 query) %f(s) [%f]\n",
        searcher.getClass().getName(), numFirstMatches, queries.numRows(),
        numMatches, queries.numRows() * NUM_RESULTS, searcherAvgTime, bruteSearchAvgTime);

    assertEquals("Closest vector returned doesn't match", queries.numRows(), numFirstMatches);
    assertTrue("Searcher " + searcher.getClass().getName() + " slower than brute",
        bruteSearchAvgTime > searcherAvgTime);
  }

  public static Pair<List<List<WeightedThing<Vector>>>, Long> getResultsAndRuntime(Searcher searcher,
                                                                                   Iterable<? extends Vector> queries) {
    long start = System.currentTimeMillis();
    List<List<WeightedThing<Vector>>> results = searcher.search(queries, NUM_RESULTS);
    long end = System.currentTimeMillis();
    return new Pair<List<List<WeightedThing<Vector>>>, Long>(results, end - start);
  }

  public static Pair<List<WeightedThing<Vector>>, Long> getResultsAndRuntimeSearchFirst(
      Searcher searcher, Iterable<? extends Vector> queries) {
    long start = System.currentTimeMillis();
    List<WeightedThing<Vector>> results = searcher.searchFirst(queries, false);
    long end = System.currentTimeMillis();
    return new Pair<List<WeightedThing<Vector>>, Long>(results, end - start);
  }

  static class StripWeight implements Function<WeightedThing<Vector>, Vector> {
    @Override
    public Vector apply(WeightedThing<Vector> input) {
      Preconditions.checkArgument(input != null, "input is null");
      //noinspection ConstantConditions
      return input.getValue();
    }
  }
}
