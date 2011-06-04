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
package org.apache.mahout.df.mapreduce.partial;

import java.io.IOException;

import com.google.common.io.Closeables;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.df.node.Node;

import com.google.common.base.Preconditions;

/**
 * Stores/Loads the intermediate results of step1 needed by step2.<br>
 * This class should not be needed outside of the partial package, so all its methods are protected.<br>
 */
public final class InterResults {
  private InterResults() { }
  
  /**
   * Load the trees and the keys returned from the first step
   * 
   * @param fs
   *          forest path file system
   * @param forestPath
   *          file path to the (key,tree) file
   * @param numMaps
   *          number of map tasks
   * @param numTrees
   *          total number of trees in the forest
   * @param partition
   *          current partition
   * @param keys
   *          array of size numTrees, will contain the loaded keys
   * @param trees
   *          array of size numTrees, will contain the loaded trees
   * @return number of instances in the current partition
   * @throws IOException
   */
  public static int load(FileSystem fs,
                         Path forestPath,
                         int numMaps,
                         int numTrees,
                         int partition,
                         TreeID[] keys,
                         Node[] trees) throws IOException {
    Preconditions.checkArgument(keys.length == trees.length, "keys.length != trees.length");

    FSDataInputStream in = fs.open(forestPath);

    TreeID key = new TreeID();
    int numInstances = -1;

    try {
      // get current partition's size
      for (int p = 0; p < numMaps; p++) {
        if (p == partition) {
          numInstances = in.readInt();
        } else {
          in.readInt();
        }
      }

      // load (key, tree)
      int current = 0;
      for (int index = 0; index < numTrees; index++) {
        key.readFields(in);

        if (key.partition() == partition) {
          // skip the trees of the current partition
          Node.read(in);
        } else {
          keys[current] = key.clone();
          trees[current] = Node.read(in);
          current++;
        }
      }

      if (current != keys.length) {
        throw new IllegalStateException("loaded less keys/trees than expected");
      }
    } finally {
      Closeables.closeQuietly(in);
    }

    return numInstances;
  }
  
  /**
   * Write the forest trees into a file
   * 
   * @param fs
   *          File System
   * @param keys
   *          keys returned by the first step
   * @param trees
   *          trees returned by the first step
   * @param sizes
   *          partitions' sizes in hadoop order
   * @throws IOException
   */
  public static void store(FileSystem fs,
                           Path forestPath,
                           TreeID[] keys,
                           Node[] trees,
                           int[] sizes) throws IOException {
    Preconditions.checkArgument(keys.length == trees.length, "keys.length != trees.length");
    
    int numTrees = keys.length;
    int numMaps = sizes.length;
    
    FSDataOutputStream out = fs.create(forestPath);

    try {
      // write partitions' sizes
      for (int p = 0; p < numMaps; p++) {
        out.writeInt(sizes[p]);
      }

      // write the data
      for (int index = 0; index < numTrees; index++) {
        keys[index].write(out);
        trees[index].write(out);
      }
    } finally {
      Closeables.closeQuietly(out);
    }
  }
  
}
