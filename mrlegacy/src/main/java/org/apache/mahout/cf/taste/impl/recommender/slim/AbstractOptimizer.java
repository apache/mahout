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

import java.util.Collection;
import java.util.Map.Entry;
import java.util.concurrent.Callable;

import org.apache.commons.lang.math.RandomUtils;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.jet.random.Normal;

/**
 * Common implementation for a SLIM optimizer.
 *
 */
public abstract class AbstractOptimizer implements Optimizer {

  private final RefreshHelper refreshHelper;
  private FastByIDMap<Long> IDitemMapping;
  private FastByIDMap<Integer> itemIDMapping;
  protected final DataModel dataModel;
  private double mean;
  private double stDev;
  
  protected SlimSolution slim;
  private Normal normal;
  
  public static final long BIASID = Long.MAX_VALUE - 1;

  protected AbstractOptimizer(DataModel dataModel, double mean, double stDev)
      throws TasteException {
    this.dataModel = dataModel;
    this.mean = mean;
    this.stDev = stDev;
    buildMappings();
    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        buildMappings();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
  }

  protected SlimSolution createSlimSolution(SparseColumnMatrix itemWeights) {
    return new SlimSolution(itemIDMapping, IDitemMapping, itemWeights);
  }
  
  public SlimSolution getSlimSolution() {
    return slim;
  }
    
  protected void prepareTraining() throws TasteException {
    int numItems = dataModel.getNumItems();
    this.normal = new Normal(mean, stDev,
        org.apache.mahout.common.RandomUtils.getRandom());

    SparseColumnMatrix itemWeights = new SparseColumnMatrix(numItems + 1, numItems + 1);

    slim = createSlimSolution(itemWeights);
  }

  @Override
  public synchronized double getAndInitWeightPos(Matrix itemWeights, int row, int column) {
    if (row == column)
      return 0;

    double weight = itemWeights.getQuick(row, column);
    if (weight == 0) {
      // weight = TestRandom.nextDouble();
      weight = Math.abs(normal.nextDouble());
      itemWeights.setQuick(row, column, weight);
    }
    return weight;
  }

  @Override
  public double getAndInitWeight(Matrix itemWeights, int row, int column) {
    if (row == column)
      return 0;

    double weight = itemWeights.getQuick(row, column);
    if (weight == 0) {
      weight = normal.nextDouble();
      itemWeights.setQuick(row, column, weight);
    }
    return weight;
  }

  private void buildMappings() throws TasteException {
    int numItems = dataModel.getNumItems();
    LongPrimitiveIterator it = dataModel.getItemIDs();
    itemIDMapping = createIDMapping(numItems, it);
    IDitemMapping = new FastByIDMap<Long>(dataModel.getNumItems());
    for (Entry<Long, Integer> entry : itemIDMapping.entrySet()) {
      IDitemMapping.put(entry.getValue(), entry.getKey());
    }
  }

  /**
   * Returns the item ID associated with the given index.
   */
  public long IDIndex(int itemIndex) throws NoSuchItemException {
    Long itemID = IDitemMapping.get(itemIndex);
    if (itemID == null) {
      throw new NoSuchItemException(itemIndex);
    }
    return itemID;
  }

  /**
   * Returns the index associated with itemID.
   */
  protected Integer itemIndex(long itemID) {
    Integer itemIndex = itemIDMapping.get(itemID);
    if (itemIndex == null) {
      itemIndex = itemIDMapping.size();
      itemIDMapping.put(itemID, itemIndex);
    }
    return itemIndex;
  }


  /**
   * Samples a user ID.
   */
  protected long sampleUserID() throws TasteException {
    LongPrimitiveIterator it = dataModel.getUserIDs();
    int skip;
    do {
      skip = RandomUtils.nextInt(dataModel.getNumUsers() + 1);
    } while (skip == 0);

    it.skip(skip - 1);
    return it.next();
  }

  
  /**
   * Samples an item from a user's preferences.
   */
  protected int samplePosItemIndex(PreferenceArray userItems) {
    int index = RandomUtils.nextInt(userItems.length());
    return itemIndex(userItems.getItemID(index));
  }

  /**
   * Samples an item not included in user's preferences.
   */
  protected int sampleNegItemIndex(PreferenceArray userItems)
      throws TasteException {
    int itemIndex;
    long itemID;
    do {
      itemIndex = RandomUtils.nextInt(dataModel.getNumItems() + 1);
      itemID = IDIndex(itemIndex);
    } while (userItems.hasPrefWithItemID(itemID));

    return itemIndex;
  }

  private static FastByIDMap<Integer> createIDMapping(int size,
      LongPrimitiveIterator idIterator) {
    FastByIDMap<Integer> mapping = new FastByIDMap<Integer>(size);
    int index = 0;
    mapping.put(BIASID, index++);
    while (idIterator.hasNext()) {
      mapping.put(idIterator.nextLong(), index++);
    }
    return mapping;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

}
