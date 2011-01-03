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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;

/**
 * base class for {@link Factorizer}s, provides ID to index mapping
 */
public abstract class AbstractFactorizer implements Factorizer {

  private final FastByIDMap<Integer> userIDMapping;
  private final FastByIDMap<Integer> itemIDMapping;

  protected AbstractFactorizer(DataModel dataModel) throws TasteException {
    userIDMapping = createIDMapping(dataModel.getNumUsers(), dataModel.getUserIDs());
    itemIDMapping = createIDMapping(dataModel.getNumItems(), dataModel.getItemIDs());
  }

  protected Factorization createFactorization(double[][] userFeatures, double[][] itemFeatures) {
    return new Factorization(userIDMapping, itemIDMapping, userFeatures, itemFeatures);
  }

  protected Integer userIndex(long userID) {
    return userIDMapping.get(userID);
  }

  protected Integer itemIndex(long itemID) {
    return itemIDMapping.get(itemID);
  }

  private FastByIDMap<Integer> createIDMapping(int size, LongPrimitiveIterator idIterator) {
    FastByIDMap<Integer> mapping = new FastByIDMap<Integer>(size);
    int index = 0;
    while (idIterator.hasNext()) {
      mapping.put(idIterator.nextLong(), index++);
    }
    return mapping;
  }
}
