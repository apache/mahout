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

import com.google.common.base.Preconditions;
import org.apache.hadoop.io.LongWritable;

/**
 * Indicates both the tree and the data partition used to grow the tree
 */
public class TreeID extends LongWritable implements Cloneable {
  
  public static final int MAX_TREEID = 100000;
  
  public TreeID() { }
  
  public TreeID(int partition, int treeId) {
    Preconditions.checkArgument(partition >= 0, "Wrong partition: " + partition + ". Partition must be >= 0!");
    Preconditions.checkArgument(treeId >= 0, "Wrong treeId: " + treeId + ". TreeId must be >= 0!");
    set(partition, treeId);
  }
  
  public void set(int partition, int treeId) {
    set((long) partition * MAX_TREEID + treeId);
  }
  
  /**
   * Data partition (InputSplit's index) that was used to grow the tree
   */
  public int partition() {
    return (int) (get() / MAX_TREEID);
  }
  
  public int treeId() {
    return (int) (get() % MAX_TREEID);
  }
  
  @Override
  public TreeID clone() {
    return new TreeID(partition(), treeId());
  }
}
