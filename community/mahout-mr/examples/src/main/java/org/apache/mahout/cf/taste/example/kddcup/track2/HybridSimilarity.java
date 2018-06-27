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

package org.apache.mahout.cf.taste.example.kddcup.track2;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.similarity.AbstractItemSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

final class HybridSimilarity extends AbstractItemSimilarity {

  private final ItemSimilarity cfSimilarity;
  private final ItemSimilarity contentSimilarity;

  HybridSimilarity(DataModel dataModel, File dataFileDirectory) throws IOException {
    super(dataModel);
    cfSimilarity = new LogLikelihoodSimilarity(dataModel);
    contentSimilarity = new TrackItemSimilarity(dataFileDirectory);
  }

  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    return contentSimilarity.itemSimilarity(itemID1, itemID2) * cfSimilarity.itemSimilarity(itemID1, itemID2);
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    double[] result = contentSimilarity.itemSimilarities(itemID1, itemID2s);
    double[] multipliers = cfSimilarity.itemSimilarities(itemID1, itemID2s);
    for (int i = 0; i < result.length; i++) {
      result[i] *= multipliers[i];
    }
    return result;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    cfSimilarity.refresh(alreadyRefreshed);
  }

}
