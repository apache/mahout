
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
public final class VectorBinaryAssignCostTest {
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
    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, dense, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, dense, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThisLookupThat.class,
        VectorBinaryAssign.getBestOperation(dense, dense, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllLoopInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, dense, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, dense, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void sasvInteractions() {
    replayAll();

    assertEquals(VectorBinaryAssign.AssignIterateUnionSequentialMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, sasv, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignIterateUnionSequentialMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, sasv, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignIterateUnionSequentialMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, sasv, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllIterateSequentialMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, sasv, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignIterateUnionSequentialMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, sasv, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void rasvInteractions() {
    replayAll();

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, rasv, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, rasv, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThisLookupThat.class,
        VectorBinaryAssign.getBestOperation(rasv, rasv, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllLoopInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, rasv, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, rasv, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void sasvDenseInteractions() {
    replayAll();

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, dense, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, dense, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThisLookupThat.class,
        VectorBinaryAssign.getBestOperation(sasv, dense, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllIterateThisLookupThatMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, dense, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, dense, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void denseSasvInteractions() {
    replayAll();
    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, sasv, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, sasv, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignIterateUnionSequentialInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, sasv, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, sasv, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, sasv, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void denseRasvInteractions() {
    replayAll();
    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, rasv, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, rasv, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThisLookupThat.class,
        VectorBinaryAssign.getBestOperation(dense, rasv, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllLoopInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, rasv, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(dense, rasv, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void rasvDenseInteractions() {
    replayAll();
    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, dense, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, dense, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThisLookupThat.class,
        VectorBinaryAssign.getBestOperation(rasv, dense, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllLoopInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, dense, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, dense, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void sasvRasvInteractions() {
    replayAll();
    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, rasv, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, rasv, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThisLookupThat.class,
        VectorBinaryAssign.getBestOperation(sasv, rasv, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllIterateThisLookupThatMergeUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, rasv, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(sasv, rasv, Functions.SECOND_LEFT_ZERO).getClass());
  }

  @Test
  public void rasvSasvInteractions() {
    replayAll();
    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, sasv, Functions.PLUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, sasv, Functions.MINUS).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThisLookupThat.class,
        VectorBinaryAssign.getBestOperation(rasv, sasv, Functions.MULT).getClass());

    assertEquals(VectorBinaryAssign.AssignAllIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, sasv, Functions.DIV).getClass());

    assertEquals(VectorBinaryAssign.AssignNonzerosIterateThatLookupThisInplaceUpdates.class,
        VectorBinaryAssign.getBestOperation(rasv, sasv, Functions.SECOND_LEFT_ZERO).getClass());
  }


  private void replayAll() {
    replay(dense, sasv, rasv);
  }
}
