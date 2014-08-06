/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.jblas

import org.apache.mahout.math.Vector
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.AbstractVector
import org.apache.mahout.math.OrderedIntDoubleMapping

import org.jblas.DoubleMatrix

import java.util.Iterator

class JBlasVector(size: Int) extends AbstractVector(size) {
  val jm: DoubleMatrix = new DoubleMatrix(size)

  def setQuick(index: Int, v: Double): Unit = jm.put(index, v)

  def like: Vector = new JBlasVector(size)

  def matrixLike(rows: Int, cols: Int): Matrix = new JBlasMatrix(rows, cols)

  def getQuick(index: Int): Double = jm.get(index)

  def isDense: Boolean = true

  class VectorElement extends Vector.Element {
    var eidx = 0

    def get: Double = getQuick(eidx)

    def index: Int = eidx

    def set(value: Double): Unit = {
      invalidateCachedLength
      setQuick(eidx, value)
    }
  }

  def iterateNonZero: Iterator[Vector.Element] = new Iterator[Vector.Element] {
    var idx: Int = -1
    var lookAheadIdx: Int = -1
    var element: VectorElement = new VectorElement

    def lookAhead: Unit = do {
      lookAheadIdx = lookAheadIdx + 1
    } while (lookAheadIdx < size && getQuick(lookAheadIdx) == 0.0)

    def hasNext: Boolean = {
      if (lookAheadIdx == idx)
        lookAhead
      lookAheadIdx < size
    }

    def next: Vector.Element = {
      if (lookAheadIdx == idx)
        lookAhead

      idx = lookAheadIdx
      if (idx >= size)
        throw new NoSuchElementException

      element.eidx = idx
      return element
    }

    def remove: Unit = throw new UnsupportedOperationException
  }

  def iterator: Iterator[Vector.Element] = new Iterator[Vector.Element] {
    var element: VectorElement = new VectorElement

    element.eidx = -1

    def hasNext: Boolean = element.eidx + 1 < size

    def next: Vector.Element = {
      if (element.eidx >= size)
        throw new NoSuchElementException
      element.eidx = element.eidx + 1
      element
    }

    def remove: Unit = throw new UnsupportedOperationException
  }

  def getIteratorAdvanceCost: Double = 1.0

  def getLookupCost: Double = 1.0

  def getNumNondefaultElements(): Int = size

  def isAddConstantTime: Boolean = true

  def isSequentialAccess: Boolean = true

  def mergeUpdates(updates: OrderedIntDoubleMapping): Unit =
    for (i <- 0 until updates.getNumMappings)
      setQuick(updates.getIndices()(i), updates.getValues()(i))
}
