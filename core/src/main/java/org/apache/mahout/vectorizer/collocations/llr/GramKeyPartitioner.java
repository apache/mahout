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

package org.apache.mahout.vectorizer.collocations.llr;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * Partition GramKeys based on their Gram, ignoring the secondary sort key so that all GramKeys with the same
 * gram are sent to the same partition.
 */
public class GramKeyPartitioner extends Partitioner<GramKey, Gram> {

  private static final String HASH_OFFSET_PROPERTY_NAME = "grampartitioner.hash.offset";
    
  public static void setOffsets(Configuration conf, int left) {
    conf.setInt(HASH_OFFSET_PROPERTY_NAME, left);
  }
  
  private int offset;

  @Override
  public int getPartition(GramKey key, Gram value, int numPartitions) {
    // see: http://svn.apache.org/viewvc/hadoop/mapreduce/trunk/src/java/org/apache/hadoop/mapreduce/
    //      lib/partition/BinaryPartitioner.java?revision=816664&view=markup
    int length = key.getLength() - 1;
    int right = (offset + length) % length;
    int hash = WritableComparator.hashBytes(key.getBytes(), right);
    return (hash & Integer.MAX_VALUE) % numPartitions;
  }

  public void configure(Configuration conf) {
    offset = conf.getInt(HASH_OFFSET_PROPERTY_NAME, -1);
  }
}
