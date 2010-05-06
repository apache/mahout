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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.DistributedSimilarity;

/**
 * for each item-vector, we compute its weight here and map out all entries with the user as key,
 * so we can create the user-vectors in the reducer
 */
public final class PreferredItemsPerUserMapper extends MapReduceBase
    implements Mapper<LongWritable,EntityPrefWritableArrayWritable,LongWritable,ItemPrefWithItemVectorWeightWritable> {

  private DistributedSimilarity distributedSimilarity;

  @Override
  public void configure(JobConf jobConf) {
    super.configure(jobConf);
    distributedSimilarity =
      ItemSimilarityJob.instantiateSimilarity(jobConf.get(ItemSimilarityJob.DISTRIBUTED_SIMILARITY_CLASSNAME));
  }

  @Override
  public void map(LongWritable item,
                  EntityPrefWritableArrayWritable userPrefsArray,
                  OutputCollector<LongWritable,ItemPrefWithItemVectorWeightWritable> output,
                  Reporter reporter) throws IOException {

    EntityPrefWritable[] userPrefs = userPrefsArray.getPrefs();

    double weight = distributedSimilarity.weightOfItemVector(new UserPrefsIterator(userPrefs));

    for (EntityPrefWritable userPref : userPrefs) {
      output.collect(new LongWritable(userPref.getID()),
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
