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
package org.apache.mahout.common.commandline;

import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;

public final class MinhashOptionCreator {

  public static final String NUM_HASH_FUNCTIONS = "numHashFunctions";
  public static final String KEY_GROUPS = "keyGroups";
  public static final String HASH_TYPE = "hashType";
  public static final String MIN_CLUSTER_SIZE = "minClusterSize";
  public static final String MIN_VECTOR_SIZE = "minVectorSize";
  public static final String NUM_REDUCERS = "numReducers";
  public static final String DEBUG_OUTPUT = "debugOutput";

  private MinhashOptionCreator() {
  }

  public static DefaultOptionBuilder debugOutputOption() {
    return new DefaultOptionBuilder()
        .withLongName(DEBUG_OUTPUT)
        .withShortName("debug")
        .withArgument(
            new ArgumentBuilder().withName(DEBUG_OUTPUT).withDefault("false")
                .withMinimum(1).withMaximum(1).create())
        .withDescription("Cluster the whole vectors for debugging");
  }

  public static DefaultOptionBuilder numReducersOption() {
    return new DefaultOptionBuilder()
        .withLongName(NUM_REDUCERS)
        .withRequired(false)
        .withShortName("r")
        .withArgument(
            new ArgumentBuilder().withName(NUM_REDUCERS).withDefault("2")
                .withMinimum(1).withMaximum(1).create())
        .withDescription("The number of reduce tasks. Defaults to 2");
  }

  /**
   * Returns a default command line option for specifying the minimum cluster
   * size in MinHash clustering
   */
  public static DefaultOptionBuilder minClusterSizeOption() {
    return new DefaultOptionBuilder()
        .withLongName(MIN_CLUSTER_SIZE)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName(MIN_CLUSTER_SIZE).withDefault("10")
                .withMinimum(1).withMaximum(1).create())
        .withDescription("Minimum points inside a cluster")
        .withShortName("mcs");
  }

  /**
   * Returns a default command line option for specifying the type of hash to
   * use in MinHash clustering: Should one out of
   * ("linear","polynomial","murmur")
   */
  public static DefaultOptionBuilder hashTypeOption() {
    return new DefaultOptionBuilder()
        .withLongName(HASH_TYPE)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName(HASH_TYPE).withDefault("murmur")
                .withMinimum(1).withMaximum(1).create())
        .withDescription(
            "Type of hash function to use. Available types: (linear, polynomial, murmur) ")
        .withShortName("ht");
  }

  /**
   * Returns a default command line option for specifying the min size of the
   * vector to hash Should one out of ("linear","polynomial","murmur")
   */
  public static DefaultOptionBuilder minVectorSizeOption() {
    return new DefaultOptionBuilder()
        .withLongName(MIN_VECTOR_SIZE)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName(MIN_VECTOR_SIZE).withDefault("5")
                .withMinimum(1).withMaximum(1).create())
        .withDescription("Minimum size of vector to be hashed")
        .withShortName("mvs");
  }

  /**
   * Returns a default command line option for specifying the number of hash
   * functions to be used in MinHash clustering
   */
  public static DefaultOptionBuilder numHashFunctionsOption() {
    return new DefaultOptionBuilder()
        .withLongName(NUM_HASH_FUNCTIONS)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName(NUM_HASH_FUNCTIONS)
                .withDefault("10").withMinimum(1).withMaximum(1).create())
        .withDescription("Number of hash functions to be used")
        .withShortName("nh");
  }

  /**
   * Returns a default command line option for specifying the number of key
   * groups to be used in MinHash clustering
   */
  public static DefaultOptionBuilder keyGroupsOption() {
    return new DefaultOptionBuilder()
        .withLongName(KEY_GROUPS)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName(KEY_GROUPS).withDefault("2")
                .withMinimum(1).withMaximum(1).create())
        .withDescription("Number of key groups to be used").withShortName("kg");
  }
}
