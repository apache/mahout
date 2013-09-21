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

package org.apache.mahout.math;

import static org.junit.Assert.assertEquals;

import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

@RunWith(Parameterized.class)
public final class VectorBinaryAggregateTest {
  private static final int CARDINALITY = 1000;

  private final DoubleDoubleFunction aggregator;
  private final DoubleDoubleFunction combiner;
  private final VectorBinaryAggregate operation;
  private final Vector first;
  private final Vector second;

  Random r = new Random();

  @SuppressWarnings("unchecked")
  @Parameterized.Parameters
  public static Collection<Object[]> generateData() {
    List<Object[]> data = Lists.newArrayList();
    for (List<?> entry : Sets.cartesianProduct(Lists.newArrayList(
        ImmutableSet.of(Functions.PLUS, Functions.PLUS_ABS, Functions.MAX),
        ImmutableSet.of(Functions.PLUS, Functions.PLUS_ABS, Functions.MULT, Functions.MULT_RIGHT_PLUS1,
            Functions.MINUS),
        ImmutableSet.copyOf(VectorBinaryAggregate.OPERATIONS),
        ImmutableSet.of(new SequentialAccessSparseVector(CARDINALITY),
          new RandomAccessSparseVector(CARDINALITY),
          new DenseVector(CARDINALITY)),
        ImmutableSet.of(new SequentialAccessSparseVector(CARDINALITY),
          new RandomAccessSparseVector(CARDINALITY),
          new DenseVector(CARDINALITY))))) {
      data.add(entry.toArray());
    }
    return data;
  }

  public VectorBinaryAggregateTest(DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner,
                                   VectorBinaryAggregate operation, Vector first, Vector second) {
    this.aggregator = aggregator;
    this.combiner = combiner;
    this.operation = operation;
    this.first = first;
    this.second = second;
  }

  @Test
  public void testSelf() {
    Vector x = first.like();
    Vector xBase = new DenseVector(CARDINALITY);
    List<Double> items = Lists.newArrayList();
    for (int i = 0; i < x.size(); ++i) {
      items.add(r.nextDouble());
    }

    for (int i = 1; i < x.size(); ++i) {
      x.setQuick(i, items.get(i));
      xBase.setQuick(i, items.get(i));
    }

    Vector y = second.like().assign(x);
    Vector yBase = new DenseVector(x);

    System.out.printf("aggregator %s; combiner %s; operation %s\n", aggregator, combiner, operation);
    double expectedResult = combiner.apply(0, 0);
    for (int i = 1; i < x.size(); ++i) {
      expectedResult = aggregator.apply(expectedResult, combiner.apply(items.get(i), items.get(i)));
    }

    double result = operation.aggregate(x, y, aggregator, combiner);
    double resultBase = operation.aggregate(xBase, yBase, aggregator, combiner);

    assertEquals(expectedResult, result, 0.0);
    assertEquals(resultBase, result, 0.0);
  }

  @Test
  public void testSeparate() {
    List<Double> items1 = Lists.newArrayList();
    List<Double> items2 = Lists.newArrayList();
    for (int i = 0; i < CARDINALITY; ++i) {
      items1.add(r.nextDouble());
      items2.add(r.nextDouble());
    }

    Vector x = first.like();
    Vector xBase = new DenseVector(CARDINALITY);
    for (int i = 0; i < x.size(); ++i) {
      x.setQuick(i, items1.get(i));
      xBase.setQuick(i, items1.get(i));
    }

    Vector y = second.like();
    Vector yBase = new DenseVector(CARDINALITY);
    for (int i = 0; i < y.size(); ++i) {
      y.setQuick(i, items2.get(i));
      yBase.setQuick(i, items2.get(i));
    }


    System.out.printf("aggregator %s; combiner %s; operation %s\n", aggregator, combiner, operation);
    double expectedResult = combiner.apply(items1.get(0), items2.get(0));
    for (int i = 1; i < x.size(); ++i) {
      expectedResult = aggregator.apply(expectedResult, combiner.apply(items1.get(i), items2.get(i)));
    }

    double result = operation.aggregate(x, y, aggregator, combiner);
    double resultBase = operation.aggregate(xBase, yBase, aggregator, combiner);

    assertEquals(expectedResult, result, 0.0);
    assertEquals(resultBase, result, 0.0);
  }
}
