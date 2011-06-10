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

package org.apache.mahout.common;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;

public final class DummyRecordWriter<K, V> extends RecordWriter<K, V> {

  private final Map<K, List<V>> data = new TreeMap<K, List<V>>();

  @Override
  public void write(K key, V value) {
    List<V> points = data.get(key);
    if (points == null) {
      points = Lists.newArrayList();
      data.put(key, points);
    }
    points.add(value);
  }

  @Override
  public void close(TaskAttemptContext context) {
  }

  public Map<K, List<V>> getData() {
    return data;
  }

  public List<V> getValue(K key) {
    return data.get(key);
  }

  public Set<K> getKeys() {
    return data.keySet();
  }

  public static <K1, V1, K2, V2> Mapper<K1, V1, K2, V2>.Context build(Mapper<K1, V1, K2, V2> mapper,
                                                                      Configuration configuration,
                                                                      RecordWriter<K2, V2> output)
    throws IOException, InterruptedException {
    return mapper.new Context(configuration, new TaskAttemptID(), null, output, null, new DummyStatusReporter(), null);
  }

  public static <K1, V1, K2, V2> Reducer<K1, V1, K2, V2>.Context build(Reducer<K1, V1, K2, V2> reducer,
                                                                       Configuration configuration,
                                                                       RecordWriter<K2, V2> output,
                                                                       Class<K1> keyClass,
                                                                       Class<V1> valueClass)
    throws IOException, InterruptedException {
    return reducer.new Context(configuration,
                               new TaskAttemptID(),
                               new MockIterator(),
                               null,
                               null,
                               output,
                               null,
                               new DummyStatusReporter(),
                               null,
                               keyClass,
                               valueClass);
  }

}
