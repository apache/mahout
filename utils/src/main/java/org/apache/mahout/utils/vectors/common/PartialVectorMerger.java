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

package org.apache.mahout.utils.vectors.common;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.mahout.math.VectorWritable;

/**
 * This class groups a set of input vectors. The Sequence file input should have
 * a {@link WritableComparable} key containing document id and a
 * {@link VectorWritable} value containing the term frequency vector. This class
 * also does normalization of the vector.
 * 
 */
public final class PartialVectorMerger {
  
  public static final float NO_NORMALIZING = -1.0f;
  
  public static final String NORMALIZATION_POWER = "normalization.power";
  
  /**
   * Cannot be initialized. Use the static functions
   */
  private PartialVectorMerger() {

  }
  
  /**
   * Merge all the partial
   * {@link org.apache.mahout.math.RandomAccessSparseVector}s into the complete
   * Document {@link org.apache.mahout.math.RandomAccessSparseVector}
   * 
   * @param partialVectorPaths
   *          input directory of the vectors in {@link SequenceFile} format
   * @param output
   *          output directory were the partial vectors have to be created
   * @param normPower
   *          The normalization value. Must be greater than or equal to 0 or
   *          equal to {@link #NO_NORMALIZING}
   * @throws IOException
   */
  public static void mergePartialVectors(List<Path> partialVectorPaths,
                                         String output,
                                         float normPower) throws IOException {
    if (normPower != NO_NORMALIZING && normPower < 0) {
      throw new IllegalArgumentException("normPower must either be -1 or >= 0");
    }
    
    Configurable client = new JobClient();
    JobConf conf = new JobConf(PartialVectorMerger.class);
    conf.set("io.serializations",
      "org.apache.hadoop.io.serializer.JavaSerialization,"
          + "org.apache.hadoop.io.serializer.WritableSerialization");
    // this conf parameter needs to be set enable serialisation of conf values
    conf.setJobName("PartialVectorMerger::MergePartialVectors");
    
    conf.setFloat(NORMALIZATION_POWER, normPower);
    
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(VectorWritable.class);
    
    FileInputFormat.setInputPaths(conf,
      getCommaSeparatedPaths(partialVectorPaths));
    
    Path outputPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outputPath);
    
    conf.setMapperClass(IdentityMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setReducerClass(PartialVectorMergeReducer.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    
    FileSystem dfs = FileSystem.get(outputPath.toUri(), conf);
    if (dfs.exists(outputPath)) {
      dfs.delete(outputPath, true);
    }
    
    client.setConf(conf);
    JobClient.runJob(conf);
  }
  
  private static String getCommaSeparatedPaths(List<Path> paths) {
    StringBuilder commaSeparatedPaths = new StringBuilder();
    String sep = "";
    for (Path path : paths) {
      commaSeparatedPaths.append(sep).append(path.toString());
      sep = ",";
    }
    return commaSeparatedPaths.toString();
  }
}
