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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.DistributedItemSimilarity;
import org.apache.mahout.common.iterator.IteratorIterable;
import org.apache.mahout.math.VarLongWritable;

/**
 * for each item-vector, we compute its weight here and map out all entries with the user as key,
 * so we can create the user-vectors in the reducer
 */
public final class PreferredItemsPerUserMapper extends
    Mapper<VarLongWritable,EntityPrefWritableArrayWritable,VarLongWritable,ItemPrefWithItemVectorWeightWritable> {

  private DistributedItemSimilarity distributedSimilarity;

  @Override
  protected void setup(Context context) {
    Configuration jobConf = context.getConfiguration();
    distributedSimilarity =
      ItemSimilarityJob.instantiateSimilarity(jobConf.get(ItemSimilarityJob.DISTRIBUTED_SIMILARITY_CLASSNAME));
  }

  @Override
  protected void map(VarLongWritable item,
                     EntityPrefWritableArrayWritable userPrefsArray,
                     Context context) throws IOException, InterruptedException {

    EntityPrefWritable[] userPrefs = userPrefsArray.getPrefs();

    double weight = distributedSimilarity.weightOfItemVector(
        new IteratorIterable<Float>(new UserPrefsIterator(userPrefs)));

    for (EntityPrefWritable userPref : userPrefs) {
      context.write(new VarLongWritable(userPref.getID()),
          new ItemPrefWithItemVectorWeightWritable(item.get(), weight, userPref.getPrefValue()));
    }
  }

  public static class UserPrefsIterator implements Iterator<Float> {

    private int index;
    private final EntityPrefWritable[] userPrefs;

    public UserPrefsIterator(EntityPrefWritable[] userPrefs) {
      this.userPrefs = userPrefs;
      this.index = 0;
    }

    @Override
    public boolean hasNext() {
      return (index < userPrefs.length);
    }

    @Override
    public Float next() {
      if (index >= userPrefs.length) {
        throw new NoSuchElementException();
      }
      return userPrefs[index++].getPrefValue();
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }

  }

}
