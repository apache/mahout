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

package org.apache.mahout.cf.taste.hadoop.item;

import com.google.common.collect.Lists;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.util.List;

/**
 * we use a neat little trick to explicitly filter items for some users: we inject a NaN summand into the preference
 * estimation for those items, which makes {@link org.apache.mahout.cf.taste.hadoop.item.AggregateAndRecommendReducer}
 * automatically exclude them 
 */
public class ItemFilterAsVectorAndPrefsReducer
    extends Reducer<VarLongWritable,VarLongWritable,VarIntWritable,VectorAndPrefsWritable> {

  private final VarIntWritable itemIDIndexWritable = new VarIntWritable();
  private final VectorAndPrefsWritable vectorAndPrefs = new VectorAndPrefsWritable();

  @Override
  protected void reduce(VarLongWritable itemID, Iterable<VarLongWritable> values, Context ctx)
    throws IOException, InterruptedException {
    
    int itemIDIndex = TasteHadoopUtils.idToIndex(itemID.get());
    Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
    /* artificial NaN summand to exclude this item from the recommendations for all users specified in userIDs */
    vector.set(itemIDIndex, Double.NaN);

    List<Long> userIDs = Lists.newArrayList();
    List<Float> prefValues = Lists.newArrayList();
    for (VarLongWritable userID : values) {
      userIDs.add(userID.get());
      prefValues.add(1.0f);
    }

    itemIDIndexWritable.set(itemIDIndex);
    vectorAndPrefs.set(vector, userIDs, prefValues);
    ctx.write(itemIDIndexWritable, vectorAndPrefs);
  }
}
