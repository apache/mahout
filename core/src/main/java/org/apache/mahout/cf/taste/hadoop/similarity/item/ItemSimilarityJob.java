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
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.ToUserPrefsMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.CoRating;
import org.apache.mahout.cf.taste.hadoop.similarity.DistributedItemSimilarity;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;

/**
 * <p>Runs a completely distributed computation of the similarity of the itemvectors of the user-item-matrix
 *  as a series of mapreduces.</p>
 *
 * <p>Algorithm used is a slight modification from the algorithm described in
 * http://www.umiacs.umd.edu/~jimmylin/publications/Elsayed_etal_ACL2008_short.pdf</p>
 *
 * <pre>
 * Example using cosine distance:
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
 * <li>--similarityClassname (classname): an implemenation of {@link DistributedItemSimilarity} used to compute the
 * similarity</li>
 * <li>--maxSimilaritiesPerItem (integer): try to cap the number of similar items per item to this number
 * (default: 100)</li>
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

  public static final String DISTRIBUTED_SIMILARITY_CLASSNAME =
      "org.apache.mahout.cf.taste.hadoop.similarity.item.ItemSimilarityJob.distributedSimilarityClassname";

  public static final String NUMBER_OF_USERS =
      "org.apache.mahout.cf.taste.hadoop.similarity.item.ItemSimilarityJob.numberOfUsers";

  public static final String MAX_SIMILARITIES_PER_ITEM =
      "org.apache.mahout.cf.taste.hadoop.similarity.item.ItemSimilarityJob.maxSimilaritiesPerItem";

  private static final Integer DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM = 100;

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    addInputOption();
    addOutputOption();
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate");
    addOption("maxSimilaritiesPerItem", "m", "try to cap the number of similar items per item to this number " +
    		"(default: " + DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM + ")", String.valueOf(DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM));

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    String distributedSimilarityClassname = parsedArgs.get("--similarityClassname");
    int maxSimilaritiesPerItem = Integer.parseInt(parsedArgs.get("--maxSimilaritiesPerItem"));

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Path tempDirPath = new Path(parsedArgs.get("--tempDir"));

    Path countUsersPath = new Path(tempDirPath, "countUsers");
    Path itemVectorsPath = new Path(tempDirPath, "itemVectors");
    Path userVectorsPath = new Path(tempDirPath, "userVectors");
    Path similaritiesPath = new Path(tempDirPath, "similarities");
    Path cappedSimilaritiesPath = new Path(tempDirPath, "cappedSimilarities");

    AtomicInteger currentPhase = new AtomicInteger();

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      /* count all unique users */
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
      Job itemVectors = prepareJob(inputPath,
                                   itemVectorsPath,
                                   TextInputFormat.class,
                                   ToUserPrefsMapper.class,
                                   VarLongWritable.class,
                                   EntityPrefWritable.class,
                                   ToItemVectorReducer.class,
                                   VarLongWritable.class,
                                   EntityPrefWritableArrayWritable.class,
                                   SequenceFileOutputFormat.class);
      itemVectors.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job userVectors = prepareJob(itemVectorsPath,
                                   userVectorsPath,
                                   SequenceFileInputFormat.class,
                                   PreferredItemsPerUserMapper.class,
                                   VarLongWritable.class,
                                   ItemPrefWithItemVectorWeightWritable.class,
                                   PreferredItemsPerUserReducer.class,
                                   VarLongWritable.class,
                                   ItemPrefWithItemVectorWeightArrayWritable.class,
                                   SequenceFileOutputFormat.class);
      userVectors.getConfiguration().set(DISTRIBUTED_SIMILARITY_CLASSNAME, distributedSimilarityClassname);
      userVectors.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job similarity = prepareJob(userVectorsPath,
                                  similaritiesPath,
                                  SequenceFileInputFormat.class,
                                  CopreferredItemsMapper.class,
                                  ItemPairWritable.class,
                                  CoRating.class,
                                  SimilarityReducer.class,
                                  EntityEntityWritable.class,
                                  DoubleWritable.class,
                                  SequenceFileOutputFormat.class);
      Configuration conf = similarity.getConfiguration();
      int numberOfUsers = readNumberOfUsers(conf, countUsersPath);
      conf.set(DISTRIBUTED_SIMILARITY_CLASSNAME, distributedSimilarityClassname);
      conf.setInt(NUMBER_OF_USERS, numberOfUsers);
      similarity.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job capSimilaritiesPerItem = prepareJob(similaritiesPath,
                                              cappedSimilaritiesPath,
                                              SequenceFileInputFormat.class,
                                              CapSimilaritiesPerItemMapper.class,
                                              CapSimilaritiesPerItemKeyWritable.class,
                                              SimilarItemWritable.class,
                                              CapSimilaritiesPerItemReducer.class,
                                              EntityEntityWritable.class,
                                              DoubleWritable.class,
                                              SequenceFileOutputFormat.class);

      capSimilaritiesPerItem.getConfiguration().setInt(MAX_SIMILARITIES_PER_ITEM, maxSimilaritiesPerItem);
      capSimilaritiesPerItem.setPartitionerClass(
          CapSimilaritiesPerItemKeyWritable.CapSimilaritiesPerItemKeyPartitioner.class);
      capSimilaritiesPerItem.setGroupingComparatorClass(
          CapSimilaritiesPerItemKeyWritable.CapSimilaritiesPerItemKeyGroupingComparator.class);
      capSimilaritiesPerItem.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job removeDuplicates = prepareJob(cappedSimilaritiesPath,
                                        outputPath,
                                        SequenceFileInputFormat.class,
                                        Mapper.class,
                                        EntityEntityWritable.class,
                                        DoubleWritable.class,
                                        RemoveDuplicatesReducer.class,
                                        EntityEntityWritable.class,
                                        DoubleWritable.class,
                                        TextOutputFormat.class);
      removeDuplicates.waitForCompletion(true);
    }

    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ItemSimilarityJob(), args);
  }

  static int readNumberOfUsers(Configuration conf, Path outputDir) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    Path outputFile = fs.listStatus(outputDir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        return path.getName().startsWith("part-");
      }
    })[0].getPath();
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

  static DistributedItemSimilarity instantiateSimilarity(String classname) {
    try {
      return (DistributedItemSimilarity) Class.forName(classname).newInstance();
    } catch (ClassNotFoundException cnfe) {
      throw new IllegalStateException(cnfe);
    } catch (InstantiationException ie) {
      throw new IllegalStateException(ie);
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    }
  }

}
