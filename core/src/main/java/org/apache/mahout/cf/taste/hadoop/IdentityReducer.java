/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/** Copied from Hadoop 0.19. Replace when Hadoop 0.20+ makes Reducer non-abstract. */
public class IdentityReducer<K, V> extends Reducer<K, V, K, V> {

  @Override
  protected void reduce(K key, Iterable<V> values, Context context
  ) throws IOException, InterruptedException {
    for (V value : values) {
      context.write(key, value);
    }
  }

}
