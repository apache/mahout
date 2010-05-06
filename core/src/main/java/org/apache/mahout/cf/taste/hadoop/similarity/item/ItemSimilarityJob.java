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

import org.apache.commons.cli2.Option;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.ToUserPrefsMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.CoRating;
import org.apache.mahout.cf.taste.hadoop.similarity.DistributedSimilarity;
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
 * <li>--similarityClassname (classname): an implemenation of {@link DistributedSimilarity} used to compute the
 * similarity</li>
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

  @Override
  public int run(String[] args) throws IOException {

    Option similarityClassOpt = AbstractJob.buildOption("similarityClassname", "s",
    "Name of distributed similarity class to instantiate");

    Map<String,String> parsedArgs = AbstractJob.parseArguments(args, similarityClassOpt);
    if (parsedArgs == null) {
      return -1;
    }

    Configuration originalConf = getConf();

    String distributedSimilarityClassname = parsedArgs.get("--similarityClassname");

    String inputPath = originalConf.get("mapred.input.dir");
    String outputPath = originalConf.get("mapred.output.dir");
    String tempDirPath = parsedArgs.get("--tempDir");

    String itemVectorsPath = tempDirPath + "/itemVectors";
    String userVectorsPath = tempDirPath + "/userVectors";

    JobConf itemVectors = prepareJobConf(inputPath,
                                         itemVectorsPath,
                                         TextInputFormat.class,
                                         ToUserPrefsMapper.class,
                                         LongWritable.class,
                                         EntityPrefWritable.class,
                                         ToItemVectorReducer.class,
                                         LongWritable.class,
                                         EntityPrefWritableArrayWritable.class,
                                         SequenceFileOutputFormat.class);
    JobClient.runJob(itemVectors);

    JobConf userVectors = prepareJobConf(itemVectorsPath,
                                         userVectorsPath,
                                         SequenceFileInputFormat.class,
                                         PreferredItemsPerUserMapper.class,
                                         LongWritable.class,
                                         ItemPrefWithItemVectorWeightWritable.class,
                                         PreferredItemsPerUserReducer.class,
                                         LongWritable.class,
                                         ItemPrefWithItemVectorWeightArrayWritable.class,
                                         SequenceFileOutputFormat.class);

    userVectors.set(DISTRIBUTED_SIMILARITY_CLASSNAME, distributedSimilarityClassname);
    JobClient.runJob(userVectors);

    JobConf similarity = prepareJobConf(userVectorsPath,
                                        outputPath,
                                        SequenceFileInputFormat.class,
                                        CopreferredItemsMapper.class,
                                        ItemPairWritable.class,
                                        CoRating.class,
                                        SimilarityReducer.class,
                                        EntityEntityWritable.class,
                                        DoubleWritable.class,
                                        TextOutputFormat.class);

    similarity.set(DISTRIBUTED_SIMILARITY_CLASSNAME, distributedSimilarityClassname);
    JobClient.runJob(similarity);

    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ItemSimilarityJob(), args);
  }

  static DistributedSimilarity instantiateSimilarity(String classname) {
    try {
      return (DistributedSimilarity) Class.forName(classname).newInstance();
    } catch (ClassNotFoundException cnfe) {
      throw new IllegalStateException(cnfe);
    } catch (InstantiationException ie) {
      throw new IllegalStateException(ie);
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    }
  }

}
