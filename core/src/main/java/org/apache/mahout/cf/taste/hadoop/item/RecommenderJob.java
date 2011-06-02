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
import org.apache.mahout.cf.taste.hadoop.MaybePruneRowsMapper;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CountUsersKeyWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CountUsersMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CountUsersReducer;
import org.apache.mahout.cf.taste.hadoop.similarity.item.ToItemVectorsReducer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.SimilarityType;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * <p>Runs a completely distributed recommender job as a series of mapreduces.</p>
 *
 * <p>Preferences in the input file should look like {@code userID,itemID[,preferencevalue]}</p>
 *
 * <p>
 * Preference value is optional to accommodate applications that have no notion of a preference value (that is, the user
 * simply expresses a preference for an item, but no degree of preference).
 * </p>
 *
 * <p>
 * The preference value is assumed to be parseable as a {@code double}. The user IDs and item IDs are
 * parsed as {@code long}s.
 * </p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing one or more text files with the preference data</li>
 * <li>-Dmapred.output.dir=(path): output path where recommender output should go</li>
 * <li>--similarityClassname (classname): Name of distributed similarity class to instantiate or a predefined similarity
 *  from {@link SimilarityType}</li>
 * <li>--usersFile (path): only compute recommendations for user IDs contained in this file (optional)</li>
 * <li>--itemsFile (path): only include item IDs from this file in the recommendations (optional)</li>
 * <li>--filterFile (path): file containing comma-separated userID,itemID pairs. Used to exclude the item from the
 * recommendations for that user (optional)</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (10)</li>
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>
 * <li>--maxPrefsPerUser (integer): Maximum number of preferences considered per user in
 *  final recommendation phase (10)</li>
 * <li>--maxSimilaritiesPerItem (integer): Maximum number of similarities considered per item (100)</li>
 * <li>--maxCooccurrencesPerItem (integer): Maximum number of cooccurrences considered per item (100)</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class RecommenderJob extends AbstractJob {

  public static final String BOOLEAN_DATA = "booleanData";
  
  private static final int DEFAULT_MAX_SIMILARITIES_PER_ITEM = 100;
  private static final int DEFAULT_MAX_COOCCURRENCES_PER_ITEM = 100;
  private static final int DEFAULT_MIN_PREFS_PER_USER = 1;

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    addInputOption();
    addOutputOption();
    addOption("numRecommendations", "n", "Number of recommendations per user",
        String.valueOf(AggregateAndRecommendReducer.DEFAULT_NUM_RECOMMENDATIONS));
    addOption("usersFile", "u", "File of users to recommend for", null);
    addOption("itemsFile", "i", "File of items to recommend for", null);
    addOption("filterFile", "f", "File containing comma-separated userID,itemID pairs. Used to exclude the item from "
        + "the recommendations for that user (optional)", null);
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("maxPrefsPerUser", "mp",
        "Maximum number of preferences considered per user in final recommendation phase",
        String.valueOf(UserVectorSplitterMapper.DEFAULT_MAX_PREFS_PER_USER_CONSIDERED));
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this in the similarity computation "
        + "(default: " + DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MIN_PREFS_PER_USER));
    addOption("maxSimilaritiesPerItem", "m", "Maximum number of similarities considered per item ",
        String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ITEM));
    addOption("maxCooccurrencesPerItem", "mo", "try to cap the number of cooccurrences per item to this "
        + "number (default: " + DEFAULT_MAX_COOCCURRENCES_PER_ITEM + ')',
        String.valueOf(DEFAULT_MAX_COOCCURRENCES_PER_ITEM));
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate, alternatively use "
        + "one of the predefined similarities (" + SimilarityType.listEnumNames() + ')',
        String.valueOf(SimilarityType.SIMILARITY_COOCCURRENCE));    

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    int numRecommendations = Integer.parseInt(parsedArgs.get("--numRecommendations"));
    String usersFile = parsedArgs.get("--usersFile");
    String itemsFile = parsedArgs.get("--itemsFile");
    String filterFile = parsedArgs.get("--filterFile");
    boolean booleanData = Boolean.valueOf(parsedArgs.get("--booleanData"));
    int maxPrefsPerUser = Integer.parseInt(parsedArgs.get("--maxPrefsPerUser"));
    int minPrefsPerUser = Integer.parseInt(parsedArgs.get("--minPrefsPerUser"));
    int maxSimilaritiesPerItem = Integer.parseInt(parsedArgs.get("--maxSimilaritiesPerItem"));
    int maxCooccurrencesPerItem = Integer.parseInt(parsedArgs.get("--maxCooccurrencesPerItem"));
    String similarityClassname = parsedArgs.get("--similarityClassname");

    Path userVectorPath = getTempPath("userVectors");
    Path itemIDIndexPath = getTempPath("itemIDIndex");
    Path countUsersPath = getTempPath("countUsers");
    Path itemUserMatrixPath = getTempPath("itemUserMatrix");
    Path similarityMatrixPath = getTempPath("similarityMatrix");
    Path prePartialMultiplyPath1 = getTempPath("prePartialMultiply1");
    Path prePartialMultiplyPath2 = getTempPath("prePartialMultiply2");
    Path explicitFilterPath = getTempPath("explicitFilterPath");
    Path partialMultiplyPath = getTempPath("partialMultiply");

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
      toUserVector.getConfiguration().setInt(ToUserVectorReducer.MIN_PREFERENCES_PER_USER, minPrefsPerUser);
      toUserVector.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job countUsers = prepareJob(userVectorPath,
                                  countUsersPath,
                                  SequenceFileInputFormat.class,
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
      Job maybePruneAndTransponse = prepareJob(userVectorPath,
                                  itemUserMatrixPath,
                                  SequenceFileInputFormat.class,
                                  MaybePruneRowsMapper.class,
                                  IntWritable.class,
                                  DistributedRowMatrix.MatrixEntryWritable.class,
                                  ToItemVectorsReducer.class,
                                  IntWritable.class,
                                  VectorWritable.class,
                                  SequenceFileOutputFormat.class);
      maybePruneAndTransponse.getConfiguration().setInt(MaybePruneRowsMapper.MAX_COOCCURRENCES,
          maxCooccurrencesPerItem);
      maybePruneAndTransponse.waitForCompletion(true);
    }

    int numberOfUsers = TasteHadoopUtils.readIntFromFile(getConf(), countUsersPath);

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      /* Once DistributedRowMatrix uses the hadoop 0.20 API, we should refactor this call to something like
       * new DistributedRowMatrix(...).rowSimilarity(...) */
      try {
        ToolRunner.run(getConf(), new RowSimilarityJob(), new String[] {
          "-Dmapred.input.dir=" + itemUserMatrixPath,
          "-Dmapred.output.dir=" + similarityMatrixPath,
          "--numberOfColumns", String.valueOf(numberOfUsers),
          "--similarityClassname", similarityClassname,
          "--maxSimilaritiesPerRow", String.valueOf(maxSimilaritiesPerItem + 1),
          "--tempDir", getTempPath().toString() });
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
                                                    maxPrefsPerUser);
      prePartialMultiply2.waitForCompletion(true);

      Job partialMultiply = prepareJob(
          new Path(prePartialMultiplyPath1 + "," + prePartialMultiplyPath2), partialMultiplyPath,
          SequenceFileInputFormat.class, Mapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
          ToVectorAndPrefReducer.class, VarIntWritable.class, VectorAndPrefsWritable.class,
          SequenceFileOutputFormat.class);
      setS3SafeCombinedInputPath(partialMultiply, getTempPath(), prePartialMultiplyPath1, prePartialMultiplyPath2);
      partialMultiply.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {

      /* convert the user/item pairs to filter if a filterfile has been specified */
      if (filterFile != null) {
        Job itemFiltering = prepareJob(new Path(filterFile), explicitFilterPath, TextInputFormat.class,
            ItemFilterMapper.class, VarLongWritable.class, VarLongWritable.class,
            ItemFilterAsVectorAndPrefsReducer.class, VarIntWritable.class, VectorAndPrefsWritable.class,
            SequenceFileOutputFormat.class);
        itemFiltering.waitForCompletion(true);
      }

      String aggregateAndRecommendInput = partialMultiplyPath.toString();
      if (filterFile != null) {
        aggregateAndRecommendInput += "," + explicitFilterPath;
      }

      Job aggregateAndRecommend = prepareJob(
          new Path(aggregateAndRecommendInput), outputPath, SequenceFileInputFormat.class,
          PartialMultiplyMapper.class, VarLongWritable.class, PrefAndSimilarityColumnWritable.class,
          AggregateAndRecommendReducer.class, VarLongWritable.class, RecommendedItemsWritable.class,
          TextOutputFormat.class);
      Configuration aggregateAndRecommendConf = aggregateAndRecommend.getConfiguration();
      if (itemsFile != null) {
        aggregateAndRecommendConf.set(AggregateAndRecommendReducer.ITEMS_FILE, itemsFile);
      }

      if (filterFile != null) {
        setS3SafeCombinedInputPath(aggregateAndRecommend, getTempPath(), partialMultiplyPath, explicitFilterPath);
      }
      setIOSort(aggregateAndRecommend);
      aggregateAndRecommendConf.set(AggregateAndRecommendReducer.ITEMID_INDEX_PATH, itemIDIndexPath.toString());
      aggregateAndRecommendConf.setInt(AggregateAndRecommendReducer.NUM_RECOMMENDATIONS, numRecommendations);
      aggregateAndRecommendConf.setBoolean(BOOLEAN_DATA, booleanData);
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
