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

package org.apache.mahout.cf.taste.hadoop.similarity.item.writables;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.Writable;

/**
 * An {@link ArrayWritable} holding {@link ItemPrefWithLengthWritable}s
 *
 * Used as user-vector
 */
public final class ItemPrefWithLengthArrayWritable extends ArrayWritable {

  public ItemPrefWithLengthArrayWritable() {
    super(ItemPrefWithLengthWritable.class);
  }

  public ItemPrefWithLengthArrayWritable(ItemPrefWithLengthWritable[] itemPrefs) {
    super(ItemPrefWithLengthWritable.class, itemPrefs);
  }

  public ItemPrefWithLengthWritable[] getItemPrefs() {
    Writable[] writables = get();
    ItemPrefWithLengthWritable[] itemPrefs = new ItemPrefWithLengthWritable[writables.length];
    for (int n=0; n<writables.length; n++) {
      itemPrefs[n] = (ItemPrefWithLengthWritable)writables[n];
    }
    return itemPrefs;
  }
}
