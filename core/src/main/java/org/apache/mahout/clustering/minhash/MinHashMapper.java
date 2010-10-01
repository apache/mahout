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

package org.apache.mahout.clustering.minhash;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.minhash.HashFactory.HashType;
import org.apache.mahout.common.commandline.MinhashOptionCreator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class MinHashMapper extends Mapper<Text, Writable, Text, Writable> {

  private static final Logger log = LoggerFactory.getLogger(MinHashMapper.class);

  private HashFunction[] hashFunction;
  private int numHashFunctions;
  private int keyGroups;
  private int minVectorSize;
  private boolean debugOutput;
  private int[] minHashValues;
  private byte[] bytesToHash;

  @Override
  protected void setup(Context context) throws IOException,  InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    this.numHashFunctions = conf.getInt(MinhashOptionCreator.NUM_HASH_FUNCTIONS, 10);
    this.minHashValues = new int[numHashFunctions];
    this.bytesToHash = new byte[4];
    this.keyGroups = conf.getInt(MinhashOptionCreator.KEY_GROUPS, 1);
    this.minVectorSize = conf.getInt(MinhashOptionCreator.MIN_VECTOR_SIZE, 5);
    String htype = conf.get(MinhashOptionCreator.HASH_TYPE, "linear");
    this.debugOutput = conf.getBoolean(MinhashOptionCreator.DEBUG_OUTPUT, false);

    HashType hashType;
    try {
      hashType = HashType.valueOf(htype);
    } catch (IllegalArgumentException iae) {
      log.warn("No valid hash type found in configuration for {}, assuming type: {}", htype, HashType.LINEAR);
      hashType = HashType.LINEAR;
    }
    hashFunction = HashFactory.createHashFunctions(hashType, numHashFunctions);
  }

  /**
   * Hash all items with each function and retain min. value for each iteration.
   * We up with X number of minhash signatures.
   * 
   * Now depending upon the number of key-groups (1 - 4) concatenate that many
   * minhash values to form cluster-id as 'key' and item-id as 'value'
   */
  @Override
  public void map(Text item, Writable features, Context context) throws IOException, InterruptedException {
    Vector featureVector = ((VectorWritable) features).get();
    if (featureVector.size() < minVectorSize) {
      return;
    }
    // Initialize the minhash values to highest
    for (int i = 0; i < numHashFunctions; i++) {
      minHashValues[i] = Integer.MAX_VALUE;
    }
    for (int i = 0; i < numHashFunctions; i++) {
      for (Vector.Element ele : featureVector) {
        int value = (int) ele.get();
        bytesToHash[0] = (byte) (value >> 24);
        bytesToHash[1] = (byte) (value >> 16);
        bytesToHash[2] = (byte) (value >> 8);
        bytesToHash[3] = (byte) (value);
        int hashIndex = hashFunction[i].hash(bytesToHash);
        if (minHashValues[i] > hashIndex) {
          minHashValues[i] = hashIndex;
        }
      }
    }
    // output the cluster information
    for (int i = 0; i < numHashFunctions; i += keyGroups) {
      StringBuilder clusterIdBuilder = new StringBuilder();
      for (int j = 0; j < keyGroups && (i + j) < numHashFunctions; j++) {
        clusterIdBuilder.append(minHashValues[i + j]).append('-');
      }
      String clusterId = clusterIdBuilder.toString();
      clusterId = clusterId.substring(0, clusterId.lastIndexOf('-'));
      Text cluster = new Text(clusterId);
      Writable point;
      if (debugOutput) {
        point = new VectorWritable(featureVector.clone());
      } else {
        point = new Text(item.toString());
      }
      context.write(cluster, point);
    }
  }
}
