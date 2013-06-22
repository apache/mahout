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
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

@Deprecated
public class MinHashMapper extends Mapper<Text, VectorWritable, Text, Writable> {

  private HashFunction[] hashFunction;
  private int numHashFunctions;
  private int keyGroups;
  private int minVectorSize;
  private boolean debugOutput;
  private int[] minHashValues;
  private byte[] bytesToHash;
  private boolean hashValue;

  private final Text cluster = new Text();
  private final VectorWritable vector = new VectorWritable();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    numHashFunctions = conf.getInt(MinHashDriver.NUM_HASH_FUNCTIONS, 10);
    minHashValues = new int[numHashFunctions];
    bytesToHash = new byte[4];
    keyGroups = conf.getInt(MinHashDriver.KEY_GROUPS, 1);
    minVectorSize = conf.getInt(MinHashDriver.MIN_VECTOR_SIZE, 5);
    debugOutput = conf.getBoolean(MinHashDriver.DEBUG_OUTPUT, false);

    String dimensionToHash = conf.get(MinHashDriver.VECTOR_DIMENSION_TO_HASH);
    hashValue = MinHashDriver.HASH_DIMENSION_VALUE.equalsIgnoreCase(dimensionToHash);

    HashType hashType = HashType.valueOf(conf.get(MinHashDriver.HASH_TYPE));
    hashFunction = HashFactory.createHashFunctions(hashType, numHashFunctions);
  }

  /**
   * Hash all items with each function and retain min. value for each iteration. We up with X number of
   * minhash signatures.
   * <p/>
   * Now depending upon the number of key-groups (1 - 4) concatenate that many minhash values to form
   * cluster-id as 'key' and item-id as 'value'
   */
  @Override
  public void map(Text item, VectorWritable features, Context context) throws IOException, InterruptedException {
    Vector featureVector = features.get();
    if (featureVector.size() < minVectorSize) {
      return;
    }
    // Initialize the MinHash values to highest
    for (int i = 0; i < numHashFunctions; i++) {
      minHashValues[i] = Integer.MAX_VALUE;
    }

    for (int i = 0; i < numHashFunctions; i++) {
      for (Vector.Element ele : featureVector.nonZeroes()) {
        int value = hashValue ? (int) ele.get() : ele.index();
        bytesToHash[0] = (byte) (value >> 24);
        bytesToHash[1] = (byte) (value >> 16);
        bytesToHash[2] = (byte) (value >> 8);
        bytesToHash[3] = (byte) value;
        int hashIndex = hashFunction[i].hash(bytesToHash);
        //if our new hash value is less than the old one, replace the old one
        if (minHashValues[i] > hashIndex) {
          minHashValues[i] = hashIndex;
        }
      }
    }
    // output the cluster information
    for (int i = 0; i < numHashFunctions; i++) {
      StringBuilder clusterIdBuilder = new StringBuilder();
      for (int j = 0; j < keyGroups; j++) {
        clusterIdBuilder.append(minHashValues[(i + j) % numHashFunctions]).append('-');
      }
      //remove the last dash
      clusterIdBuilder.deleteCharAt(clusterIdBuilder.length() - 1);

      cluster.set(clusterIdBuilder.toString());

      if (debugOutput) {
        vector.set(featureVector);
        context.write(cluster, vector);
      } else {
        context.write(cluster, item);
      }
    }
  }
}
