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
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.UserWritable;

public final class PreferredItemsPerUserReducer
    extends Reducer<UserWritable,ItemPrefWithLengthWritable,UserWritable,ItemPrefWithLengthArrayWritable> {

  @Override
  protected void reduce(UserWritable user, Iterable<ItemPrefWithLengthWritable> itemPrefs, Context context)
      throws IOException, InterruptedException {

    Set<ItemPrefWithLengthWritable> itemPrefsWithLength = new HashSet<ItemPrefWithLengthWritable>();

    for (ItemPrefWithLengthWritable itemPrefWithLength : itemPrefs) {
      itemPrefsWithLength.add(itemPrefWithLength.deepCopy());
    }

    context.write(user, new ItemPrefWithLengthArrayWritable(
        itemPrefsWithLength.toArray(new ItemPrefWithLengthWritable[itemPrefsWithLength.size()])));
  }


}
