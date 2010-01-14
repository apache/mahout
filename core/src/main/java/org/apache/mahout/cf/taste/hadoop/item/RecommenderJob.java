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

package org.apache.mahout.cf.taste.hadoop.item;

import org.apache.commons.cli2.Option;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapFileOutputFormat;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.AbstractJob;
import org.apache.mahout.cf.taste.hadoop.ItemPrefWritable;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Map;

/**
 * Runs a completely distributed recommender job as a series of mapreduces.</p>
 *
 * <p>Command line arguments are:</p>
 *
 * <ol>
 *  <li>numRecommendations: Number of recommendations to compute per user (optional; default 10)</li>
 *  <li>input: Directory containing a text file containing user IDs
 *   for which recommendations should be computed, one per line</li>
 *  <li>output: output path where recommender output should go</li>
 *  <li>jarFile: JAR file containing implementation code</li>
 *  <li>tempDir: directory in which to place intermediate data files (optional; default "temp")</li>
 *  <li>usersFile: file containing user IDs to recommend for (optional)</li>
 * </ol>
 *
 * @see org.apache.mahout.cf.taste.hadoop.pseudo.RecommenderJob
 */
public final class RecommenderJob extends AbstractJob {

  @Override
  public int run(String[] args) throws IOException {

    Option numReccomendationsOpt = buildOption("numRecommendations", "n", "Number of recommendations per user", "10");
    Option usersFileOpt = buildOption("usersFile", "n", "Number of recommendations per user", null);

    Map<String,String> parsedArgs = parseArguments(args, numReccomendationsOpt, usersFileOpt);

    String inputPath = parsedArgs.get("--input");
    String tempDirPath = parsedArgs.get("--tempDir");
    String outputPath = parsedArgs.get("--output");
    String jarFile = parsedArgs.get("--jarFile");
    int recommendationsPerUser = Integer.parseInt(parsedArgs.get("--numRecommendations"));
    String usersFile = parsedArgs.get("--usersFile");

    String userVectorPath = tempDirPath + "/userVectors";
    String itemIDIndexPath = tempDirPath + "/itemIDIndex";
    String cooccurrencePath = tempDirPath + "/cooccurrence";

    JobConf itemIDIndexConf = prepareJobConf(inputPath,
                                             itemIDIndexPath,
                                             jarFile,
                                             TextInputFormat.class,
                                             ItemIDIndexMapper.class,
                                             IntWritable.class,
                                             LongWritable.class,
                                             ItemIDIndexReducer.class,
                                             IntWritable.class,
                                             LongWritable.class,
                                             MapFileOutputFormat.class);
    JobClient.runJob(itemIDIndexConf);

    JobConf toUserVectorConf = prepareJobConf(inputPath,
                                              userVectorPath,
                                              jarFile,
                                              TextInputFormat.class,
                                              ToItemPrefsMapper.class,
                                              LongWritable.class,
                                              ItemPrefWritable.class,
                                              ToUserVectorReducer.class,
                                              LongWritable.class,
                                              VectorWritable.class,
                                              SequenceFileOutputFormat.class);
    JobClient.runJob(toUserVectorConf);

    JobConf toCooccurrenceConf = prepareJobConf(userVectorPath,
                                                cooccurrencePath,
                                                jarFile,
                                                SequenceFileInputFormat.class,
                                                UserVectorToCooccurrenceMapper.class,
                                                IntWritable.class,
                                                IntWritable.class,
                                                UserVectorToCooccurrenceReducer.class,
                                                IntWritable.class,
                                                VectorWritable.class,
                                                MapFileOutputFormat.class);
    JobClient.runJob(toCooccurrenceConf);

    JobConf recommenderConf = prepareJobConf(userVectorPath,
                                             outputPath,
                                             jarFile,
                                             SequenceFileInputFormat.class,
                                             RecommenderMapper.class,
                                             LongWritable.class,
                                             RecommendedItemsWritable.class,
                                             IdentityReducer.class,
                                             LongWritable.class,
                                             RecommendedItemsWritable.class,
                                             TextOutputFormat.class);
    recommenderConf.set(RecommenderMapper.COOCCURRENCE_PATH, cooccurrencePath);
    recommenderConf.set(RecommenderMapper.ITEMID_INDEX_PATH, itemIDIndexPath);    
    recommenderConf.setInt(RecommenderMapper.RECOMMENDATIONS_PER_USER, recommendationsPerUser);
    recommenderConf.set(RecommenderMapper.USERS_FILE, usersFile);
    recommenderConf.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);
    JobClient.runJob(recommenderConf);
    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RecommenderJob(), args);
  }

}
