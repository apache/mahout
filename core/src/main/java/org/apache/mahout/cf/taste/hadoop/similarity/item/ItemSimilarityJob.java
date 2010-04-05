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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.EntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPairWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthWritable;
import org.apache.mahout.common.AbstractJob;

/**
 * <p>Runs a completely distributed computation of the cosine distance of the itemvectors of the user-item-matrix
 *  as a series of mapreduces.</p>
 *
 * <p>Algorithm used is a slight modification from the algorithm described in
 * http://www.umiacs.umd.edu/~jimmylin/publications/Elsayed_etal_ACL2008_short.pdf</p>
 *
 * <pre>
 * Example:
 *
 * user-item-matrix:
 *
 *                  Game   Mouse    PC
 *          Peter     0       1      2
 *          Paul      1       0      1
 *
 * Input:
 *
 *  (Peter,Mouse,1)
 *  (Peter,PC,2)
 *  (Paul,Game,1)
 *  (Paul,PC,1)
 *
 * Step 1: Create the item-vectors
 *
 *  Game  -> (Paul,1)
 *  Mouse -> (Peter,1)
 *  PC    -> (Peter,2),(Paul,1)
 *
 * Step 2: Compute the length of the item vectors, store it with the item, create the user-vectors
 *
 *  Peter -> (Mouse,1,1),(PC,2.236,2)
 *  Paul  -> (Game,1,1),(PC,2.236,2)
 *
 * Step 3: Compute the pairwise cosine for all item pairs that have been co-rated by at least one user
 *
 *  Mouse,PC  -> 1 * 2 / (1 * 2.236)
 *  Game,PC   -> 1 * 1 / (1 * 2.236)
 *
 * </pre>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing a text file containing the entries of the user-item-matrix in
 * the form userID,itemID,preference
 * computed, one per line</li>
 * <li>-Dmapred.output.dir=(path): output path where the computations output should go</li>
 * </ol>
 *
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 * <p>Please consider supplying a --tempDir parameter for this job, as is needs to write some intermediate files</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class ItemSimilarityJob extends AbstractJob {

  @Override
  public int run(String[] args) throws Exception {

    Map<String,String> parsedArgs = AbstractJob.parseArguments(args);

    if (parsedArgs == null) {
      return -1;
    }

    Configuration originalConf = getConf();
    String inputPath = originalConf.get("mapred.input.dir");
    String outputPath = originalConf.get("mapred.output.dir");
    String tempDirPath = parsedArgs.get("--tempDir");

    String itemVectorsPath = tempDirPath + "/itemVectors";
    String userVectorsPath = tempDirPath + "/userVectors";

    Job itemVectors = createJob(originalConf, "itemVectors", inputPath, itemVectorsPath, UserPrefsPerItemMapper.class,
        EntityWritable.class, EntityPrefWritable.class, ToItemVectorReducer.class, EntityWritable.class,
        EntityPrefWritableArrayWritable.class, TextInputFormat.class, SequenceFileOutputFormat.class, true);

    itemVectors.waitForCompletion(true);

    Job userVectors = createJob(originalConf, "userVectors", itemVectorsPath, userVectorsPath,
        PreferredItemsPerUserMapper.class, EntityWritable.class, ItemPrefWithLengthWritable.class,
        PreferredItemsPerUserReducer.class, EntityWritable.class, ItemPrefWithLengthArrayWritable.class);

    userVectors.waitForCompletion(true);

    Job similarity = createJob(originalConf, "similarity", userVectorsPath, outputPath,
        CopreferredItemsMapper.class, ItemPairWritable.class, FloatWritable.class, CosineSimilarityReducer.class,
        EntityEntityWritable.class, DoubleWritable.class, SequenceFileInputFormat.class, TextOutputFormat.class, false);

    similarity.waitForCompletion(true);

    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new ItemSimilarityJob(), args);
  }

  protected static Job createJob(Configuration conf,
                                 String jobName,
                                 String inputPath,
                                 String outputPath,
                                 Class<? extends Mapper> mapperClass,
                                 Class<? extends Writable> mapKeyOutClass,
                                 Class<? extends Writable> mapValueOutClass,
                                 Class<? extends Reducer> reducerClass,
                                 Class<? extends Writable> keyOutClass,
                                 Class<? extends Writable> valueOutClass) throws IOException {
    return createJob(conf, jobName, inputPath, outputPath, mapperClass, mapKeyOutClass,
        mapValueOutClass, reducerClass, keyOutClass, valueOutClass, SequenceFileInputFormat.class,
        SequenceFileOutputFormat.class, true);
  }

  protected static Job createJob(Configuration conf,
                                 String jobName,
                                 String inputPath,
                                 String outputPath,
                                 Class<? extends Mapper> mapperClass,
                                 Class<? extends Writable> mapKeyOutClass,
                                 Class<? extends Writable> mapValueOutClass,
                                 Class<? extends Reducer> reducerClass,
                                 Class<? extends Writable> keyOutClass,
                                 Class<? extends Writable> valueOutClass,
                                 Class<? extends FileInputFormat> fileInputFormatClass,
                                 Class<? extends FileOutputFormat> fileOutputFormatClass,
                                 boolean compress) throws IOException {

    Job job = new Job(conf, jobName);

    FileSystem fs = FileSystem.get(conf);

    Path inputPathPath = new Path(inputPath).makeQualified(fs);
    Path outputPathPath = new Path(outputPath).makeQualified(fs);

    FileInputFormat.setInputPaths(job, inputPathPath);
    job.setInputFormatClass(fileInputFormatClass);

    job.setMapperClass(mapperClass);
    job.setMapOutputKeyClass(mapKeyOutClass);
    job.setMapOutputValueClass(mapValueOutClass);

    job.setReducerClass(reducerClass);
    job.setOutputKeyClass(keyOutClass);
    job.setOutputValueClass(valueOutClass);


    FileOutputFormat.setOutputPath(job, outputPathPath);
    FileOutputFormat.setCompressOutput(job, compress);
    job.setOutputFormatClass(fileOutputFormatClass);

    return job;
  }

}
