/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.classifier.bayes.mapreduce.common;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.mahout.common.StringTuple;

import com.google.common.base.Preconditions;

/**
 * ensure that features all make it into the same partition.
 */
public class FeaturePartitioner implements Partitioner<StringTuple,DoubleWritable> {
  
  @Override
  public int getPartition(StringTuple key, DoubleWritable value, int numPartitions) {
    Preconditions.checkArgument(key.length() >= 2 && key.length() <= 3, "StringTuple length out of bounds");
    String feature = key.length() == 2 ? key.stringAt(1) : key.stringAt(2);

    int length = feature.length();
    int right = 0;
    if (length > 0) {
      right = (3 + length) % length;
    }
    int hash = WritableComparator.hashBytes(feature.getBytes(), right);
    return (hash & Integer.MAX_VALUE) % numPartitions;
  }
  
  @Override
  public void configure(JobConf job) {
  /* no-op */
  }
  
}
