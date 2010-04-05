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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;

/**
 * for each item-vector, we compute its length here and map out all entries with the user as key,
 * so we can create the user-vectors in the reducer
 */
public final class PreferredItemsPerUserMapper extends MapReduceBase
    implements Mapper<LongWritable,EntityPrefWritableArrayWritable,LongWritable,ItemPrefWithLengthWritable> {

  @Override
  public void map(LongWritable item,
                  EntityPrefWritableArrayWritable userPrefsArray,
                  OutputCollector<LongWritable,ItemPrefWithLengthWritable> output,
                  Reporter reporter) throws IOException {

    EntityPrefWritable[] userPrefs = userPrefsArray.getPrefs();

    double length = 0.0;
    for (EntityPrefWritable userPref : userPrefs) {
      double value = userPref.getPrefValue();
      length += value * value;
    }

    length = Math.sqrt(length);

    for (EntityPrefWritable userPref : userPrefs) {
      output.collect(new LongWritable(userPref.getID()),
          new ItemPrefWithLengthWritable(item.get(), length, userPref.getPrefValue()));
    }

  }


}
