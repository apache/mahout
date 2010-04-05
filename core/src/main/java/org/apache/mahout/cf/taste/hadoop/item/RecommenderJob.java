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

import java.io.IOException;
import java.util.Map;

import org.apache.commons.cli2.Option;
import org.apache.hadoop.conf.Configuration;
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
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.math.VectorWritable;

/**
 * <p>Runs a completely distributed recommender job as a series of mapreduces.</p>
 * 
 * <p>Command line arguments specific to this class are:</p>
 * 
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing a text file containing user IDs for which recommendations should be
 * computed, one per line</li>
 * <li>-Dmapred.output.dir=(path): output path where recommender output should go</li>
 * <li>--usersFile (path): file containing user IDs to recommend for (optional)</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (optional; default 10)</li>
 * <li>--booleanData (boolean): Treat input data as having to pref values (false)</li>
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
  public int run(String[] args) throws IOException {
    
    Option numReccomendationsOpt = AbstractJob.buildOption("numRecommendations", "n",
      "Number of recommendations per user", "10");
    Option usersFileOpt = AbstractJob.buildOption("usersFile", "u",
      "File of users to recommend for", null);
    Option booleanDataOpt = AbstractJob.buildOption("booleanData", "b",
      "Treat input as without pref values", Boolean.FALSE.toString());

    Map<String,String> parsedArgs = AbstractJob.parseArguments(
        args, numReccomendationsOpt, usersFileOpt, booleanDataOpt);
    if (parsedArgs == null) {
      return -1;
    }
    
    Configuration originalConf = getConf();
    String inputPath = originalConf.get("mapred.input.dir");
    String outputPath = originalConf.get("mapred.output.dir");
    String tempDirPath = parsedArgs.get("--tempDir");
    int recommendationsPerUser = Integer.parseInt(parsedArgs.get("--numRecommendations"));
    String usersFile = parsedArgs.get("--usersFile");
    boolean booleanData = Boolean.valueOf(parsedArgs.get("--booleanData"));
    
    String userVectorPath = tempDirPath + "/userVectors";
    String itemIDIndexPath = tempDirPath + "/itemIDIndex";
    String cooccurrencePath = tempDirPath + "/cooccurrence";
    
    JobConf itemIDIndexConf = prepareJobConf(inputPath, itemIDIndexPath,
      TextInputFormat.class, ItemIDIndexMapper.class, IntWritable.class, LongWritable.class,
      ItemIDIndexReducer.class, IntWritable.class, LongWritable.class, MapFileOutputFormat.class);
    JobClient.runJob(itemIDIndexConf);
    
    JobConf toUserVectorConf = prepareJobConf(inputPath, userVectorPath,
      TextInputFormat.class, ToItemPrefsMapper.class, LongWritable.class,
      booleanData ? LongWritable.class : EntityPrefWritable.class,
      ToUserVectorReducer.class, LongWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
    toUserVectorConf.setBoolean(BOOLEAN_DATA, booleanData);
    JobClient.runJob(toUserVectorConf);
    
    JobConf toCooccurrenceConf = prepareJobConf(userVectorPath, cooccurrencePath,
      SequenceFileInputFormat.class, UserVectorToCooccurrenceMapper.class, IntWritable.class,
      IntWritable.class, UserVectorToCooccurrenceReducer.class, IntWritable.class, VectorWritable.class,
      MapFileOutputFormat.class);
    JobClient.runJob(toCooccurrenceConf);
    
    JobConf recommenderConf = prepareJobConf(userVectorPath, outputPath,
      SequenceFileInputFormat.class, RecommenderMapper.class, LongWritable.class,
      RecommendedItemsWritable.class, IdentityReducer.class, LongWritable.class,
      RecommendedItemsWritable.class, TextOutputFormat.class);
    recommenderConf.setBoolean(BOOLEAN_DATA, booleanData);    
    recommenderConf.set(RecommenderMapper.COOCCURRENCE_PATH, cooccurrencePath);
    recommenderConf.set(RecommenderMapper.ITEMID_INDEX_PATH, itemIDIndexPath);
    recommenderConf.setInt(RecommenderMapper.RECOMMENDATIONS_PER_USER, recommendationsPerUser);
    if (usersFile != null) {
      recommenderConf.set(RecommenderMapper.USERS_FILE, usersFile);
    }
    recommenderConf.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);
    JobClient.runJob(recommenderConf);
    return 0;
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new RecommenderJob(), args);
  }
  
}
