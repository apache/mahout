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

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.EntityWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthWritable;

/**
 * for each item-vector, we compute its length here and map out all entries with the user as key,
 * so we can create the user-vectors in the reducer
 */
public final class PreferredItemsPerUserMapper
    extends Mapper<EntityWritable, EntityPrefWritableArrayWritable,EntityWritable,ItemPrefWithLengthWritable> {

  @Override
  protected void map(EntityWritable item, EntityPrefWritableArrayWritable userPrefsArray, Context context)
      throws IOException, InterruptedException {

    EntityPrefWritable[] userPrefs = userPrefsArray.getPrefs();

    double length = 0.0;
    for (EntityPrefWritable userPref : userPrefs) {
      double value = userPref.getPrefValue();
      length += value * value;
    }

    length = Math.sqrt(length);

    for (EntityPrefWritable userPref : userPrefs) {
      context.write(new EntityWritable(userPref.getID()),
          new ItemPrefWithLengthWritable(item.getID(), length, userPref.getPrefValue()));
    }

  }


}
