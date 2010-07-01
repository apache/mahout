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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.item.ItemIDIndexMapper;
import org.apache.mahout.cf.taste.hadoop.item.ItemIDIndexReducer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob;

public final class ItemSimilarityJob extends AbstractJob {

  static final String ITEM_ID_INDEX_PATH_STR = ItemSimilarityJob.class.getName() + "itemIDIndexPathStr";
  static final String MAX_SIMILARITIES_PER_ITEM = ItemSimilarityJob.class.getName() + "maxSimilarItemsPerItem";

  private static final int DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM = 100;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ItemSimilarityJob(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate");
    addOption("maxSimilaritiesPerItem", "m", "try to cap the number of similar items per item to this number " +
        "(default: " + DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM + ')', String.valueOf(DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM));

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    String similarityClassName = parsedArgs.get("--similarityClassname");
    int maxSimilarItemsPerItem = Integer.parseInt(parsedArgs.get("--maxSimilaritiesPerItem"));

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Path tempDirPath = new Path(parsedArgs.get("--tempDir"));

    Path itemIDIndexPath = new Path(tempDirPath, "itemIDIndex");
    Path countUsersPath = new Path(tempDirPath, "countUsers");
    Path itemUserMatrixPath = new Path(tempDirPath, "itemUserMatrix");
    Path similarityMatrixPath = new Path(tempDirPath, "similarityMatrix");

    AtomicInteger currentPhase = new AtomicInteger();

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job itemIDIndex = prepareJob(
        inputPath, itemIDIndexPath, TextInputFormat.class,
        ItemIDIndexMapper.class, VarIntWritable.class, VarLongWritable.class,
        ItemIDIndexReducer.class, VarIntWritable.class, VarLongWritable.class,
        SequenceFileOutputFormat.class);
      itemIDIndex.setCombinerClass(ItemIDIndexReducer.class);
      itemIDIndex.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job countUsers = prepareJob(inputPath,
                                  countUsersPath,
                                  TextInputFormat.class,
                                  CountUsersMapper.class,
                                  CountUsersKeyWritable.class,
                                  VarLongWritable.class,
                                  CountUsersReducer.class,
                                  VarIntWritable.class,
                                  NullWritable.class,
                                  TextOutputFormat.class);
        countUsers.setPartitionerClass(CountUsersKeyWritable.CountUsersPartitioner.class);
        countUsers.setGroupingComparatorClass(CountUsersKeyWritable.CountUsersGroupComparator.class);
        countUsers.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job itemUserMatrix = prepareJob(inputPath,
                                  itemUserMatrixPath,
                                  TextInputFormat.class,
                                  PrefsToItemUserMatrixMapper.class,
                                  VarIntWritable.class,
                                  DistributedRowMatrix.MatrixEntryWritable.class,
                                  PrefsToItemUserMatrixReducer.class,
                                  IntWritable.class,
                                  VectorWritable.class,
                                  SequenceFileOutputFormat.class);
      itemUserMatrix.waitForCompletion(true);
    }

    int numberOfUsers = readNumberOfUsers(getConf(), countUsersPath);

    /* Once DistributedRowMatrix uses the hadoop 0.20 API, we should refactor this call to something like
     * new DistributedRowMatrix(...).rowSimilarity(...) */
    RowSimilarityJob.main(new String[] { "-Dmapred.input.dir=" + itemUserMatrixPath.toString(),
        "-Dmapred.output.dir=" + similarityMatrixPath.toString(), "--numberOfColumns", String.valueOf(numberOfUsers),
        "--similarityClassname", similarityClassName, "--maxSimilaritiesPerRow",
        String.valueOf(maxSimilarItemsPerItem + 1), "--tempDir", tempDirPath.toString() });

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job mostSimilarItems = prepareJob(similarityMatrixPath,
                                  outputPath,
                                  SequenceFileInputFormat.class,
                                  MostSimilarItemPairsMapper.class,
                                  EntityEntityWritable.class,
                                  DoubleWritable.class,
                                  MostSimilarItemPairsReducer.class,
                                  EntityEntityWritable.class,
                                  DoubleWritable.class,
                                  TextOutputFormat.class);
      Configuration mostSimilarItemsConf = mostSimilarItems.getConfiguration();
      mostSimilarItemsConf.set(ITEM_ID_INDEX_PATH_STR, itemIDIndexPath.toString());
      mostSimilarItemsConf.setInt(MAX_SIMILARITIES_PER_ITEM, maxSimilarItemsPerItem);
      mostSimilarItems.setCombinerClass(MostSimilarItemPairsReducer.class);
      mostSimilarItems.waitForCompletion(true);
    }

    return 0;
  }

  static int readNumberOfUsers(Configuration conf, Path outputDir) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    Path outputFile = fs.listStatus(outputDir, TasteHadoopUtils.PARTS_FILTER)[0].getPath();
    InputStream in = null;
    try  {
      in = fs.open(outputFile);
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      IOUtils.copyBytes(in, out, conf);
      return Integer.parseInt(new String(out.toByteArray(), Charset.forName("UTF-8")).trim());
    } finally {
      IOUtils.closeStream(in);
    }
  }
}
