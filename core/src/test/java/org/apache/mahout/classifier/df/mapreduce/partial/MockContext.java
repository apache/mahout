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
package org.apache.mahout.classifier.df.mapreduce.partial;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;

/**
 * Special implementation that collects the output of the mappers
 */
final class MockContext extends Context {

  private final TreeID[] keys;
  private final MapredOutput[] values;
  private int index;

  MockContext(Mapper<?,?,?,?> mapper, Configuration conf, TaskAttemptID taskid, int nbTrees)
    throws IOException, InterruptedException {
    mapper.super(conf, taskid, null, null, null, null, null);

    keys = new TreeID[nbTrees];
    values = new MapredOutput[nbTrees];
  }

  @Override
  public void write(Object key, Object value) throws IOException {
    if (index == keys.length) {
      throw new IOException("Received more output than expected : " + index);
    }

    keys[index] = ((TreeID) key).clone();
    values[index] = ((MapredOutput) value).clone();

    index++;
  }

  /**
   * @return number of outputs collected
   */
  public int nbOutputs() {
    return index;
  }

  public TreeID[] getKeys() {
    return keys;
  }

  public MapredOutput[] getValues() {
    return values;
  }
}
