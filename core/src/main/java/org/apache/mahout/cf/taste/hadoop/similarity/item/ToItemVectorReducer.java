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
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.ItemWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.UserPrefArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.UserPrefWritable;

/**
 * For each single item, collect all users with their preferences
 * (thereby building the item vectors of the user-item-matrix)
 */
public final class ToItemVectorReducer
    extends Reducer<ItemWritable,UserPrefWritable,ItemWritable,UserPrefArrayWritable> {

  @Override
  protected void reduce(ItemWritable item, Iterable<UserPrefWritable> userPrefs, Context context)
      throws IOException, InterruptedException {

    Set<UserPrefWritable> collectedUserPrefs = new HashSet<UserPrefWritable>();

    for (UserPrefWritable userPref : userPrefs) {
      collectedUserPrefs.add(userPref.deepCopy());
    }

    context.write(item, new UserPrefArrayWritable(
        collectedUserPrefs.toArray(new UserPrefWritable[collectedUserPrefs.size()])));
  }

}
