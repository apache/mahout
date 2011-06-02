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

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.MaybePruneRowsMapper;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.cf.taste.hadoop.item.ItemIDIndexMapper;
import org.apache.mahout.cf.taste.hadoop.item.ItemIDIndexReducer;
import org.apache.mahout.cf.taste.hadoop.item.RecommenderJob;
import org.apache.mahout.cf.taste.hadoop.item.ToUserVectorReducer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.SimilarityType;

/**
 * <p>Distributed precomputation of the item-item-similarities for Itembased Collaborative Filtering</p>
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
 * <li>-Dmapred.output.dir=(path): output path where similarity data should be written</li>
 * <li>--similarityClassname (classname): Name of distributed similarity class to instantiate or a predefined similarity
 *  from {@link SimilarityType}</li>
 * <li>--maxSimilaritiesPerItem (integer): Maximum number of similarities considered per item (100)</li>
 * <li>--maxCooccurrencesPerItem (integer): Maximum number of cooccurrences considered per item (100)</li>
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public final class ItemSimilarityJob extends AbstractJob {

  static final String ITEM_ID_INDEX_PATH_STR = ItemSimilarityJob.class.getName() + ".itemIDIndexPathStr";
  static final String MAX_SIMILARITIES_PER_ITEM = ItemSimilarityJob.class.getName() + ".maxSimilarItemsPerItem";

  private static final int DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM = 100;
  private static final int DEFAULT_MAX_COOCCURRENCES_PER_ITEM = 100;
  private static final int DEFAULT_MIN_PREFS_PER_USER = 1;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ItemSimilarityJob(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate, alternatively use "
        + "one of the predefined similarities (" + SimilarityType.listEnumNames() + ')');
    addOption("maxSimilaritiesPerItem", "m", "try to cap the number of similar items per item to this number "
        + "(default: " + DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM + ')',
        String.valueOf(DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM));
    addOption("maxCooccurrencesPerItem", "mo", "try to cap the number of cooccurrences per item to this number "
        + "(default: " + DEFAULT_MAX_COOCCURRENCES_PER_ITEM + ')',
        String.valueOf(DEFAULT_MAX_COOCCURRENCES_PER_ITEM));
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this "
        + "(default: " + DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MIN_PREFS_PER_USER));
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    String similarityClassName = parsedArgs.get("--similarityClassname");
    int maxSimilarItemsPerItem = Integer.parseInt(parsedArgs.get("--maxSimilaritiesPerItem"));
    int maxCooccurrencesPerItem = Integer.parseInt(parsedArgs.get("--maxCooccurrencesPerItem"));
    int minPrefsPerUser = Integer.parseInt(parsedArgs.get("--minPrefsPerUser"));
    boolean booleanData = Boolean.valueOf(parsedArgs.get("--booleanData"));

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();

    Path itemIDIndexPath = getTempPath("itemIDIndex");
    Path countUsersPath = getTempPath("countUsers");
    Path userVectorPath = getTempPath("userVectors");
    Path itemUserMatrixPath = getTempPath("itemUserMatrix");
    Path similarityMatrixPath = getTempPath("similarityMatrix");

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
      Job toUserVector = prepareJob(inputPath,
                                  userVectorPath,
                                  TextInputFormat.class,
                                  ToItemPrefsMapper.class,
                                  VarLongWritable.class,
                                  booleanData ? VarLongWritable.class : EntityPrefWritable.class,
                                  ToUserVectorReducer.class,
                                  VarLongWritable.class,
                                  VectorWritable.class,
                                  SequenceFileOutputFormat.class);
      toUserVector.getConfiguration().setBoolean(RecommenderJob.BOOLEAN_DATA, booleanData);
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
      countUsers.setCombinerClass(CountUsersCombiner.class);
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

    /* Once DistributedRowMatrix uses the hadoop 0.20 API, we should refactor this call to something like
     * new DistributedRowMatrix(...).rowSimilarity(...) */
    ToolRunner.run(getConf(), new RowSimilarityJob(), new String[] {
      "-Dmapred.input.dir=" + itemUserMatrixPath,
      "-Dmapred.output.dir=" + similarityMatrixPath,
      "--numberOfColumns", String.valueOf(numberOfUsers),
      "--similarityClassname", similarityClassName,
      "--maxSimilaritiesPerRow", String.valueOf(maxSimilarItemsPerItem + 1),
      "--tempDir", getTempPath().toString() });

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
}
