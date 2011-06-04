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

package org.apache.mahout.clustering.spectral.common;

import java.io.IOException;
import java.net.URI;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * This class handles reading and writing vectors to the Hadoop
 * distributed cache. Created as a result of Eigencuts' liberal use
 * of such functionality, but available to any algorithm requiring it.
 */
public final class VectorCache {

  private VectorCache() {
  }

  /**
   * 
   * @param key SequenceFile key
   * @param vector Vector to save, to be wrapped as VectorWritable
   */
  public static void save(Writable key,
                          Vector vector,
                          Path output,
                          Configuration conf,
                          boolean overwritePath,
                          boolean deleteOnExit) throws IOException {
    
    FileSystem fs = FileSystem.get(conf);
    output = fs.makeQualified(output);
    if (overwritePath) {
      HadoopUtil.delete(conf, output);
    }

    // set the cache
    DistributedCache.setCacheFiles(new URI[] {output.toUri()}, conf);
    
    // set up the writer
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, output, 
        IntWritable.class, VectorWritable.class);
    try {
      writer.append(key, new VectorWritable(vector));
    } finally {
      Closeables.closeQuietly(writer);
    }

    if (deleteOnExit) {
      fs.deleteOnExit(output);
    }
  }
  
  /**
   * Calls the save() method, setting the cache to overwrite any previous
   * Path and to delete the path after exiting
   */
  public static void save(Writable key, Vector vector, Path output, Configuration conf) throws IOException {
    save(key, vector, output, conf, true, true);
  }
  
  /**
   * Loads the vector from {@link DistributedCache}. Returns null if no vector exists.
   */
  public static Vector load(Configuration conf) throws IOException {
    URI[] files = DistributedCache.getCacheFiles(conf);
    if (files == null || files.length < 1) {
      return null;
    }
    return load(conf, new Path(files[0].getPath()));
  }
  
  /**
   * Loads a Vector from the specified path. Returns null if no vector exists.
   */
  public static Vector load(Configuration conf, Path input) throws IOException {
    SequenceFileValueIterator<VectorWritable> iterator =
        new SequenceFileValueIterator<VectorWritable>(input, true, conf);
    try {
      return iterator.next().get();
    } finally {
      Closeables.closeQuietly(iterator);
    }
  }
}
