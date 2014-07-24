/**
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

package org.apache.mahout.cf.taste.impl.recommender.slim;

import java.io.Serializable;
import java.util.Map;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseColumnMatrix;

import com.google.common.base.Preconditions;

/**
 * SLIM solution wrapper (W coefficient matrix).
 *
 */
public class SlimSolution implements Serializable {

  private static final long serialVersionUID = 1L;

  /**
   * Item weight matrix: W matrix in the original paper.
   */
  private final SparseColumnMatrix itemWeights;

  /**
   * Used to find the rows in the item features matrix by itemID.
   */
  private FastByIDMap<Integer> itemIDMapping;
  
  /**
   * Used to find item IDs from index.
   */
  private FastByIDMap<Long> IDitemMapping;

  public SlimSolution(FastByIDMap<Integer> itemIDMapping, FastByIDMap<Long> IDitemMapping, SparseColumnMatrix itemWeights) {
    this.itemIDMapping = Preconditions.checkNotNull(itemIDMapping);
    this.itemWeights = itemWeights;
    this.IDitemMapping = IDitemMapping;
  }
  
  public Matrix getItemWeights() {
    return itemWeights;
  }

  public int itemIndex(long itemID) throws NoSuchItemException {
    Integer index = itemIDMapping.get(itemID);
    if (index == null) {
      throw new NoSuchItemException(itemID);
    }
    return index;
  }
  
  public long IDIndex(int itemIndex) throws NoSuchItemException {
    Long itemID = IDitemMapping.get(itemIndex);
    if (itemID == null) {
      throw new NoSuchItemException(itemIndex);
    }
    return itemID;
  }

  public Iterable<Map.Entry<Long, Integer>> getItemIDMappings() {
    return itemIDMapping.entrySet();
  }

}
