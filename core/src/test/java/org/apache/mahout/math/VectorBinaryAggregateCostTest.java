
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more contributor license
 * agreements. See the NOTICE file distributed with this work for additional information regarding
 * copyright ownership. The ASF licenses this file to You under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.easymock.EasyMock.expect;
import static org.easymock.EasyMock.replay;
import static org.junit.Assert.assertEquals;

@RunWith(JUnit4.class)
public final class VectorBinaryAggregateCostTest {
  RandomAccessSparseVector realRasv = new RandomAccessSparseVector(1000000);
  SequentialAccessSparseVector realSasv = new SequentialAccessSparseVector(1000000);
  DenseVector realDense = new DenseVector(1000000);

  Vector rasv = EasyMock.createMock(Vector.class);
  Vector sasv = EasyMock.createMock(Vector.class);
  Vector dense = EasyMock.createMock(Vector.class);

  private static void createStubs(Vector v, Vector realV) {
    expect(v.getLookupCost())
        .andStubReturn(realV instanceof SequentialAccessSparseVector
            ? Math.round(Math.log(1000)) : realV.getLookupCost());
    expect(v.getIteratorAdvanceCost())
        .andStubReturn(realV.getIteratorAdvanceCost());
    expect(v.isAddConstantTime())
        .andStubReturn(realV.isAddConstantTime());
    expect(v.isSequentialAccess())
        .andStubReturn(realV.isSequentialAccess());
    expect(v.isDense())
        .andStubReturn(realV.isDense());
    expect(v.getNumNondefaultElements())
        .andStubReturn(realV.isDense() ? realV.size() : 1000);
    expect(v.size())
        .andStubReturn(realV.size());
  }

  @Before
  public void setUpStubs() {
    createStubs(dense, realDense);
    createStubs(sasv, realSasv);
    createStubs(rasv, realRasv);
  }

  @Test
  public void denseInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(dense, dense, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, dense, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, dense, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, dense, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, dense, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(dense, dense, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void sasvInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateIterateIntersection.class,
        VectorBinaryAggregate.getBestOperation(sasv, sasv, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, sasv, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, sasv, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, sasv, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, sasv, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateIterateIntersection.class,
        VectorBinaryAggregate.getBestOperation(sasv, sasv, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void rasvInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(rasv, rasv, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, rasv, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, rasv, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, rasv, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, rasv, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(rasv, rasv, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void sasvDenseInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(sasv, dense, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, dense, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, dense, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, dense, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(sasv, dense, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(sasv, dense, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void denseSasvInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThatLookupThis.class,
        VectorBinaryAggregate.getBestOperation(dense, sasv, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, sasv, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, sasv, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, sasv, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionSequential.class,
        VectorBinaryAggregate.getBestOperation(dense, sasv, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThatLookupThis.class,
        VectorBinaryAggregate.getBestOperation(dense, sasv, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void denseRasvInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThatLookupThis.class,
        VectorBinaryAggregate.getBestOperation(dense, rasv, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(dense, rasv, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(dense, rasv, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(dense, rasv, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(dense, rasv, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThatLookupThis.class,
        VectorBinaryAggregate.getBestOperation(dense, rasv, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void rasvDenseInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(rasv, dense, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, dense, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, dense, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, dense, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, dense, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(rasv, dense, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void sasvRasvInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(sasv, rasv, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(sasv, rasv, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(sasv, rasv, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(sasv, rasv, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(sasv, rasv, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThisLookupThat.class,
        VectorBinaryAggregate.getBestOperation(sasv, rasv, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }

  @Test
  public void rasvSasvInteractions() {
    replayAll();

    // Dot product
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThatLookupThis.class,
        VectorBinaryAggregate.getBestOperation(rasv, sasv, Functions.PLUS, Functions.MULT).getClass());

    // Chebyshev distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, sasv, Functions.MAX_ABS, Functions.MINUS).getClass());

    // Euclidean distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, sasv, Functions.PLUS, Functions.MINUS_SQUARED).getClass());

    // Manhattan distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, sasv, Functions.PLUS, Functions.MINUS_ABS).getClass());

    // Minkowski distance
    assertEquals(VectorBinaryAggregate.AggregateIterateUnionRandom.class,
        VectorBinaryAggregate.getBestOperation(rasv, sasv, Functions.PLUS, Functions.minusAbsPow(1.2)).getClass());

    // Tanimoto distance
    assertEquals(VectorBinaryAggregate.AggregateNonzerosIterateThatLookupThis.class,
        VectorBinaryAggregate.getBestOperation(rasv, sasv, Functions.PLUS, Functions.MULT_SQUARE_LEFT).getClass());
  }


  private void replayAll() {
    replay(dense, sasv, rasv);
  }
}
