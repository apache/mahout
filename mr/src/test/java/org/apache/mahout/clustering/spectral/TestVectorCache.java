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

package org.apache.mahout.clustering.spectral;

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
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

public class TestVectorCache extends MahoutTestCase {

  private static final double [] VECTOR = { 1, 2, 3, 4 };
  
  @Test
  public void testSave() throws Exception {
    Configuration conf = getConfiguration();
    Writable key = new IntWritable(0);
    Vector value = new DenseVector(VECTOR);
    Path path = getTestTempDirPath("output");
    
    // write the vector out
    VectorCache.save(key, value, path, conf, true, true);
    
    // can we read it from here?
    SequenceFileValueIterator<VectorWritable> iterator =
        new SequenceFileValueIterator<VectorWritable>(path, true, conf);
    try {
      VectorWritable old = iterator.next();
      // test if the values are identical
      assertEquals("Saved vector is identical to original", old.get(), value);
    } finally {
      Closeables.close(iterator, true);
    }
  }
  
  @Test
  public void testLoad() throws Exception {
    // save a vector manually
    Configuration conf = getConfiguration();
    Writable key = new IntWritable(0);
    Vector value = new DenseVector(VECTOR);
    Path path = getTestTempDirPath("output");

    FileSystem fs = FileSystem.get(path.toUri(), conf);
    // write the vector
    path = fs.makeQualified(path);
    fs.deleteOnExit(path);
    HadoopUtil.delete(conf, path);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
    try {
      writer.append(key, new VectorWritable(value));
    } finally {
      Closeables.close(writer, false);
    }
    DistributedCache.setCacheFiles(new URI[] {path.toUri()}, conf);

    // load it
    Vector result = VectorCache.load(conf);
    
    // are they the same?
    assertNotNull("Vector is null", result);
    assertEquals("Loaded vector is not identical to original", result, value);
  }
  
  @Test
  public void testAll() throws Exception {
    Configuration conf = getConfiguration();
    Vector v = new DenseVector(VECTOR);
    Path toSave = getTestTempDirPath("output");
    Writable key = new IntWritable(0);
    
    // save it
    VectorCache.save(key, v, toSave, conf);
    
    // now, load it back
    Vector v2 = VectorCache.load(conf);
    
    // are they the same?
    assertNotNull("Vector is null", v2);
    assertEquals("Vectors are not identical", v2, v);
  }
}
