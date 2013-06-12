/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *3
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.utils;


import java.io.IOException;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;

/*
 * Map-reduce job to combine two matrices A and B to (a1,a2,...aN,b1,b2,...bN)
 * Technically works on Vector files, so will also concatenate two vectors.
 * If either input is a NamedVector, the output has the name: A.name has precedence over B.name.
 * Concatenation or per-member combinations given a function object.
 * 
 * Uses clever hack which requires different matrices to have a different number of columns.
 * Courtesy of Jake Mannix, https://issues.apache.org/jira/browse/MAHOUT-884
 * If vectors are same length, this will not concatenate them in the right order
 * 
 * TODO: generalize to multiple matrices, should the teeming masses so desire
 */

public class ConcatenateVectorsJob extends AbstractJob {
  
  static final String MATRIXA_DIMS = "mahout.concatenatevectors.matrixA_dims";
  static final String MATRIXB_DIMS = "mahout.concatenatevectors.matrixB_dims";
  
  private ConcatenateVectorsJob() {}
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ConcatenateVectorsJob(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {
    addOption("matrixA", "ma", "A (left) matrix directory", true);
    addOption("matrixB", "mb", "B (right) matrix directory", true);
    addOutputOption();
    DefaultOptionCreator.overwriteOption().create();

    if (parseArguments(args) == null) {
      return -1;
    }

    Path pathA = new Path(getOption("matrixA"));
    Path pathB = new Path(getOption("matrixB"));
    Path pathOutput = getOutputPath();

    Configuration configuration = getConf();
    FileSystem fs = FileSystem.get(configuration);

    Class<? extends Writable> keyClassA = getKeyClass(pathA, fs);
    Class<? extends Writable> keyClassB = getKeyClass(pathB, fs);

    Preconditions.checkArgument(keyClassA.equals(keyClassB), "All SequenceFiles must use same key class");

    int dimA = getDimensions(pathA);
    int dimB = getDimensions(pathB);
    
    String nameA = getOption("matrixA");
    String nameB = getOption("matrixB");
    
    Job concatenate = prepareJob(
      new Path(nameA + "," + nameB), pathOutput, Mapper.class, keyClassA, VectorWritable.class,
      ConcatenateVectorsReducer.class, keyClassA, VectorWritable.class);

    configuration = concatenate.getConfiguration();
    configuration.set(MATRIXA_DIMS, Integer.toString(dimA));
    configuration.set(MATRIXB_DIMS, Integer.toString(dimB));
    // TODO: add reducer as combiner - need a system that can exercise combiners

    boolean succeeded = concatenate.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }
    return 0;
  }

  private Class<? extends Writable> getKeyClass(Path path, FileSystem fs) throws IOException {
    // this works for both part* and a directory/ with part*.
    Path pathPattern = new Path(path, "part*");
    FileStatus[] paths = fs.globStatus(pathPattern);
    Preconditions.checkArgument(paths.length == 0, path.getName() + " is a file, should be a directory");

    Path file = paths[0].getPath();
    SequenceFile.Reader reader = null;
    try {
      reader = new SequenceFile.Reader(fs, file, fs.getConf());
      return reader.getKeyClass().asSubclass(Writable.class);
    } finally {
      Closeables.close(reader, true);
    }
  }
}
