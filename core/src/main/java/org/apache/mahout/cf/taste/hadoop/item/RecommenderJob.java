/*
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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CountUsersKeyWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CountUsersMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CountUsersReducer;
import org.apache.mahout.cf.taste.hadoop.similarity.item.PrefsToItemUserMatrixMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.PrefsToItemUserMatrixReducer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob;

/**
 * <p>Runs a completely distributed recommender job as a series of mapreduces.</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing a text file containing user IDs
 *  for which recommendations should be computed, one per line</li>
 * <li>-Dmapred.output.dir=(path): output path where recommender output should go</li>
 * <li>--similarityClassname (classname): Name of distributed similarity class to instantiate</li>
 * <li>--usersFile (path): file containing user IDs to recommend for (optional)</li>
 * <li>--itemsFile (path): file containing item IDs to recommend for (optional)</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (optional; default 10)</li>
 * <li>--booleanData (boolean): Treat input data as having to pref values (false)</li>
 * <li>--maxPrefsPerUserConsidered (integer): Maximum number of preferences considered per user in
 *  final recommendation phase (10)</li>
 * <li>--maxSimilaritiesPerItemConsidered (integer): Maximum number of similarities considered per item (optional;
 *  default 100)</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class RecommenderJob extends AbstractJob {

  static final String MAX_SIMILARITIES_PER_ITEM_CONSIDERED = RecommenderJob.class.getName() +
      ".maxSimilaritiesPerItemConsidered";

  public static final String BOOLEAN_DATA = "booleanData";
  public static final int DEFAULT_MAX_SIMILARITIES_PER_ITEM_CONSIDERED = 100;

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    addInputOption();
    addOutputOption();
    addOption("numRecommendations", "n", "Number of recommendations per user",
      String.valueOf(AggregateAndRecommendReducer.DEFAULT_NUM_RECOMMENDATIONS));
    addOption("usersFile", "u", "File of users to recommend for", null);
    addOption("itemsFile", "u", "File of items to recommend for", null);
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("maxPrefsPerUserConsidered", null,
      "Maximum number of preferences considered per user in final recommendation phase",
      String.valueOf(UserVectorSplitterMapper.DEFAULT_MAX_PREFS_PER_USER_CONSIDERED));
    addOption("maxSimilaritiesPerItemConsidered", null,
      "Maximum number of similarities considered per item ",
      String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ITEM_CONSIDERED));
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate");

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Path tempDirPath = new Path(parsedArgs.get("--tempDir"));
    int numRecommendations = Integer.parseInt(parsedArgs.get("--numRecommendations"));
    String usersFile = parsedArgs.get("--usersFile");
    String itemsFile = parsedArgs.get("--itemsFile");
    boolean booleanData = Boolean.valueOf(parsedArgs.get("--booleanData"));
    int maxPrefsPerUserConsidered = Integer.parseInt(parsedArgs.get("--maxPrefsPerUserConsidered"));
    int maxSimilaritiesPerItemConsidered = Integer.parseInt(parsedArgs.get("--maxSimilaritiesPerItemConsidered"));
    String similarityClassname = parsedArgs.get("--similarityClassname");

    Path userVectorPath = new Path(tempDirPath, "userVectors");
    Path itemIDIndexPath = new Path(tempDirPath, "itemIDIndex");
    Path countUsersPath = new Path(tempDirPath, "countUsers");
    Path itemUserMatrixPath = new Path(tempDirPath, "itemUserMatrix");
    Path maybePruneItemUserMatrixPath = new Path(tempDirPath, "maybePruneItemUserMatrixPath");
    Path similarityMatrixPath = new Path(tempDirPath, "similarityMatrix");
    Path prePartialMultiplyPath1 = new Path(tempDirPath, "prePartialMultiply1");
    Path prePartialMultiplyPath2 = new Path(tempDirPath, "prePartialMultiply2");
    Path partialMultiplyPath = new Path(tempDirPath, "partialMultiply");

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
      Job toUserVector = prepareJob(
        inputPath, userVectorPath, TextInputFormat.class,
        ToItemPrefsMapper.class, VarLongWritable.class, booleanData ? VarLongWritable.class : EntityPrefWritable.class,
        ToUserVectorReducer.class, VarLongWritable.class, VectorWritable.class,
        SequenceFileOutputFormat.class);
      toUserVector.getConfiguration().setBoolean(BOOLEAN_DATA, booleanData);
      toUserVector.waitForCompletion(true);
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

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job maybePruneItemUserMatrix = prepareJob(itemUserMatrixPath,
                                  maybePruneItemUserMatrixPath,
                                  SequenceFileInputFormat.class,
                                  MaybePruneRowsMapper.class,
                                  IntWritable.class,
                                  VectorWritable.class,
                                  Reducer.class,
                                  IntWritable.class,
                                  VectorWritable.class,
                                  SequenceFileOutputFormat.class);
      maybePruneItemUserMatrix.getConfiguration().setInt(MAX_SIMILARITIES_PER_ITEM_CONSIDERED,
          maxSimilaritiesPerItemConsidered);
      maybePruneItemUserMatrix.waitForCompletion(true);
    }

    int numberOfUsers = TasteHadoopUtils.readIntFromFile(getConf(), countUsersPath);

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      /* Once DistributedRowMatrix uses the hadoop 0.20 API, we should refactor this call to something like
       * new DistributedRowMatrix(...).rowSimilarity(...) */
      try {
        RowSimilarityJob.main(new String[] { "-Dmapred.input.dir=" + maybePruneItemUserMatrixPath.toString(),
            "-Dmapred.output.dir=" + similarityMatrixPath.toString(), "--numberOfColumns",
            String.valueOf(numberOfUsers), "--similarityClassname", similarityClassname, "--maxSimilaritiesPerRow",
            String.valueOf(maxSimilaritiesPerItemConsidered + 1), "--tempDir", tempDirPath.toString() });
      } catch (Exception e) {
        throw new IllegalStateException("item-item-similarity computation failed", e);
      }
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job prePartialMultiply1 = prepareJob(
        similarityMatrixPath, prePartialMultiplyPath1, SequenceFileInputFormat.class,
        SimilarityMatrixRowWrapperMapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
        Reducer.class, VarIntWritable.class, VectorOrPrefWritable.class,
        SequenceFileOutputFormat.class);
      prePartialMultiply1.waitForCompletion(true);

      Job prePartialMultiply2 = prepareJob(
        userVectorPath, prePartialMultiplyPath2, SequenceFileInputFormat.class,
        UserVectorSplitterMapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
        Reducer.class, VarIntWritable.class, VectorOrPrefWritable.class,
        SequenceFileOutputFormat.class);
      if (usersFile != null) {
        prePartialMultiply2.getConfiguration().set(UserVectorSplitterMapper.USERS_FILE, usersFile);
      }
      prePartialMultiply2.getConfiguration().setInt(UserVectorSplitterMapper.MAX_PREFS_PER_USER_CONSIDERED,
                                                    maxPrefsPerUserConsidered);
      prePartialMultiply2.waitForCompletion(true);

      Job partialMultiply = prepareJob(
        new Path(prePartialMultiplyPath1 + "," + prePartialMultiplyPath2), partialMultiplyPath,
        SequenceFileInputFormat.class,
        Mapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
        ToVectorAndPrefReducer.class, VarIntWritable.class, VectorAndPrefsWritable.class,
        SequenceFileOutputFormat.class);
      partialMultiply.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job aggregateAndRecommend = prepareJob(
          partialMultiplyPath, outputPath, SequenceFileInputFormat.class,
          PartialMultiplyMapper.class, VarLongWritable.class, PrefAndSimilarityColumnWritable.class,
          AggregateAndRecommendReducer.class, VarLongWritable.class, RecommendedItemsWritable.class,
          TextOutputFormat.class);
      Configuration jobConf = aggregateAndRecommend.getConfiguration();
      if (itemsFile != null) {
    	  jobConf.set(AggregateAndRecommendReducer.ITEMS_FILE, itemsFile);
      }
      setIOSort(aggregateAndRecommend);
      jobConf.set(AggregateAndRecommendReducer.ITEMID_INDEX_PATH, itemIDIndexPath.toString());
      jobConf.setInt(AggregateAndRecommendReducer.NUM_RECOMMENDATIONS, numRecommendations);
      jobConf.setBoolean(BOOLEAN_DATA, booleanData);
      aggregateAndRecommend.waitForCompletion(true);
    }

    return 0;
  }

  private static void setIOSort(JobContext job) {
    Configuration conf = job.getConfiguration();
    conf.setInt("io.sort.factor", 100);
    int assumedHeapSize = 512;
    String javaOpts = conf.get("mapred.child.java.opts");
    if (javaOpts != null) {
      Matcher m = Pattern.compile("-Xmx([0-9]+)([mMgG])").matcher(javaOpts);
      if (m.find()) {
        assumedHeapSize = Integer.parseInt(m.group(1));
        String megabyteOrGigabyte = m.group(2);
        if ("g".equalsIgnoreCase(megabyteOrGigabyte)) {
          assumedHeapSize *= 1024;
        }
      }
    }
    conf.setInt("io.sort.mb", assumedHeapSize / 2);
    // For some reason the Merger doesn't report status for a long time; increase
    // timeout when running these jobs
    conf.setInt("mapred.task.timeout", 60 * 60 * 1000);
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new RecommenderJob(), args);
  }
}
