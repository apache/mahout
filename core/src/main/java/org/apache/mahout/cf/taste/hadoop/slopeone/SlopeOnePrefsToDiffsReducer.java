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

package org.apache.mahout.cf.taste.hadoop.slopeone;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.math.VarLongWritable;

public final class SlopeOnePrefsToDiffsReducer extends
    Reducer<VarLongWritable,EntityPrefWritable,EntityEntityWritable,FloatWritable> {
  
  @Override
  protected void reduce(VarLongWritable key,
                        Iterable<EntityPrefWritable> values,
                        Context context) throws IOException, InterruptedException {
    List<EntityPrefWritable> prefs = Lists.newArrayList();
    for (EntityPrefWritable writable : values) {
      prefs.add(new EntityPrefWritable(writable));
    }
    Collections.sort(prefs, ByItemIDComparator.getInstance());
    int size = prefs.size();
    for (int i = 0; i < size; i++) {
      EntityPrefWritable first = prefs.get(i);
      long itemAID = first.getID();
      float itemAValue = first.getPrefValue();
      for (int j = i + 1; j < size; j++) {
        EntityPrefWritable second = prefs.get(j);
        long itemBID = second.getID();
        float itemBValue = second.getPrefValue();
        context.write(new EntityEntityWritable(itemAID, itemBID), new FloatWritable(itemBValue - itemAValue));
      }
    }
  }
  
}