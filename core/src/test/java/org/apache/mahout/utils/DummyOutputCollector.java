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

package org.apache.mahout.utils;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.OutputCollector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class DummyOutputCollector<K extends WritableComparable, V extends Writable> implements OutputCollector<K, V> {

  final Map<String, List<V>> data = new TreeMap<String, List<V>>();

  @Override
  public void collect(K key, V values)
      throws IOException {
    List<V> points = data.get(key.toString());
    if (points == null) {
      points = new ArrayList<V>();
      data.put(key.toString(), points);
    }
    points.add(values);
  }

  public Map<String, List<V>> getData() {
    return data;
  }

  public List<V> getValue(String key) {
    return data.get(key);
  }

  public Set<String> getKeys() {
    return data.keySet();
  }

}
