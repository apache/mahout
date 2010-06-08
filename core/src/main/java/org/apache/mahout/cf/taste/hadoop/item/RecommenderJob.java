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

import org.apache.commons.cli2.Option;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
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
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;

/**
 * <p>Runs a completely distributed recommender job as a series of mapreduces.</p>
 * 
 * <p>Command line arguments specific to this class are:</p>
 * 
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing a text file containing user IDs
 *  for which recommendations should be computed, one per line</li>
 * <li>-Dmapred.output.dir=(path): output path where recommender output should go</li>
 * <li>--usersFile (path): file containing user IDs to recommend for (optional)</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (optional; default 10)</li>
 * <li>--booleanData (boolean): Treat input data as having to pref values (false)</li>
 * <li>--maxPrefsPerUserConsidered (integer): Maximum number of preferences considered per user in
 *  final recommendation phase (10)</li>
 * <li>--maxCooccurrencesPerItemConsidered: Maximum number of cooccurrences considered per item
 *  in count phase (100)</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class RecommenderJob extends AbstractJob {

  public static final String BOOLEAN_DATA = "booleanData";
  
  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    
    addOption("numRecommendations", "n", "Number of recommendations per user",
      String.valueOf(AggregateAndRecommendReducer.DEFAULT_NUM_RECOMMENDATIONS));
    addOption("usersFile", "u", "File of users to recommend for", null);
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("maxPrefsPerUserConsidered", null,
      "Maximum number of preferences considered per user in final recommendation phase",
      String.valueOf(UserVectorSplitterMapper.DEFAULT_MAX_PREFS_PER_USER_CONSIDERED));
    addOption("maxCooccurrencesPerItemConsidered", null,
      "Maximum number of cooccurrences considered per item in count phase",
      String.valueOf(UserVectorToCooccurrenceMapper.DEFAULT_MAX_COOCCURRENCES_PER_ITEM_CONSIDERED));

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    
    Configuration originalConf = getConf();
    Path inputPath = new Path(originalConf.get("mapred.input.dir"));
    Path outputPath = new Path(originalConf.get("mapred.output.dir"));
    Path tempDirPath = new Path(parsedArgs.get("--tempDir"));
    int numRecommendations = Integer.parseInt(parsedArgs.get("--numRecommendations"));
    String usersFile = parsedArgs.get("--usersFile");
    boolean booleanData = Boolean.valueOf(parsedArgs.get("--booleanData"));
    int maxPrefsPerUserConsidered = Integer.parseInt(parsedArgs.get("--maxPrefsPerUserConsidered"));
    int maxCooccurrencesPerItemConsidered = Integer.parseInt(parsedArgs.get("--maxCooccurrencesPerItemConsidered"));

    Path userVectorPath = new Path(tempDirPath, "userVectors");
    Path itemIDIndexPath = new Path(tempDirPath, "itemIDIndex");
    Path cooccurrencePath = new Path(tempDirPath, "cooccurrence");
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
      Job toCooccurrence = prepareJob(
        userVectorPath, cooccurrencePath, SequenceFileInputFormat.class,
        UserVectorToCooccurrenceMapper.class, VarIntWritable.class, VarIntWritable.class,
        UserVectorToCooccurrenceReducer.class, VarIntWritable.class, VectorWritable.class,
        SequenceFileOutputFormat.class);
      setIOSort(toCooccurrence);
      toCooccurrence.getConfiguration().setInt(UserVectorToCooccurrenceMapper.MAX_COOCCURRENCES_PER_ITEM_CONSIDERED,
                                               maxCooccurrencesPerItemConsidered);
      toCooccurrence.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job prePartialMultiply1 = prepareJob(
        cooccurrencePath, prePartialMultiplyPath1, SequenceFileInputFormat.class,
        CooccurrenceColumnWrapperMapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
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
          PartialMultiplyMapper.class, VarLongWritable.class, VectorWritable.class,
          AggregateAndRecommendReducer.class, VarLongWritable.class, RecommendedItemsWritable.class,
          TextOutputFormat.class);
      Configuration jobConf = aggregateAndRecommend.getConfiguration();
      setIOSort(aggregateAndRecommend);
      aggregateAndRecommend.setCombinerClass(AggregateCombiner.class);
      jobConf.set(AggregateAndRecommendReducer.ITEMID_INDEX_PATH, itemIDIndexPath.toString());
      jobConf.setInt(AggregateAndRecommendReducer.NUM_RECOMMENDATIONS, numRecommendations);
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
