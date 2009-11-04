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

package org.apache.mahout.df.mapred.partial;

import java.io.IOException;

import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.mapreduce.partial.TreeID;

public class PartialOutputCollector implements OutputCollector<TreeID, MapredOutput> {

  private final TreeID[] keys;

  private final MapredOutput[] values;

  private int index = 0;

  public PartialOutputCollector(int nbTrees) {
    keys = new TreeID[nbTrees];
    values = new MapredOutput[nbTrees];
  }

  public TreeID[] getKeys() {
    return keys;
  }

  public MapredOutput[] getValues() {
    return values;
  }

  @Override
  public void collect(TreeID key, MapredOutput value) throws IOException {
    if (index == keys.length) {
      throw new IOException("Received more output than expected : " + index);
    }

    keys[index] = key.clone();
    values[index] = value.clone();

    index++;
  }

  /**
   * Number of outputs collected
   * 
   * @return
   */
  public int nbOutputs() {
    return index;
  }
}
