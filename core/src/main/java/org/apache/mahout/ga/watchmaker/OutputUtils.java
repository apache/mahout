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

package org.apache.mahout.ga.watchmaker;

import java.io.IOException;
import java.util.Collection;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;

/** Utility Class that deals with the output. */
public final class OutputUtils {
  
  private OutputUtils() {
    // do nothing
  }
  
  /**
   * Lists all files in the output {@code Path}
   *
   * @param fs {@code FileSystem} to use
   * @param outpath output {@code Path}
   * @return {@code Path} array
   */
  public static Path[] listOutputFiles(FileSystem fs, Path outpath) throws IOException {
    FileStatus[] status = fs.listStatus(outpath);
    Collection<Path> outpaths = Lists.newArrayList();
    for (FileStatus s : status) {
      if (!s.isDir()) {
        outpaths.add(s.getPath());
      }
    }
    
    Path[] outfiles = new Path[outpaths.size()];
    outpaths.toArray(outfiles);
    
    return outfiles;
  }
  
  /**
   * Reads back the evaluations.
   *
   * @param outpath output {@code Path}
   * @param evaluations List of evaluations
   */
  public static void importEvaluations(FileSystem fs,
                                       Configuration conf,
                                       Path outpath,
                                       Collection<Double> evaluations) throws IOException {
    SequenceFile.Sorter sorter = new SequenceFile.Sorter(fs, LongWritable.class, DoubleWritable.class, conf);
    
    // merge and sort the outputs
    Path[] outfiles = listOutputFiles(fs, outpath);
    Path output = new Path(outpath, "output.sorted");
    sorter.merge(outfiles, output);
    
    // import the evaluations
    for (DoubleWritable value : new SequenceFileValueIterable<DoubleWritable>(output, true, conf)) {
      evaluations.add(value.get());
    }
  }
  
}
