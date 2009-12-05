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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.mahout.cf.taste.hadoop.AbstractJob;
import org.apache.mahout.cf.taste.hadoop.ItemPrefWritable;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.matrix.Vector;

import java.util.Map;

public final class RecommenderJob extends AbstractJob {

  public static void main(String[] args) throws Exception {

    Map<String,Object> parsedArgs = parseArguments(args);

    String prefsFile = parsedArgs.get("input").toString();
    String outputPath = parsedArgs.get("output").toString();
    String jarFile = parsedArgs.get("jarFile").toString();
    String userVectorPath = outputPath + "/userVectors";
    String cooccurrencePath = outputPath + "/cooccurrence";

    JobConf toUserVectorConf = prepareJobConf(prefsFile,
                                              userVectorPath,
                                              jarFile,
                                              TextInputFormat.class,
                                              ToItemPrefsMapper.class,
                                              LongWritable.class,
                                              ItemPrefWritable.class,
                                              ToUserVectorReducer.class,
                                              LongWritable.class,
                                              Vector.class,
                                              SequenceFileOutputFormat.class);
    JobClient.runJob(toUserVectorConf);

    JobConf toCooccurrenceConf = prepareJobConf(userVectorPath,
                                                cooccurrencePath,
                                                jarFile,
                                                SequenceFileInputFormat.class,
                                                UserVectorToCooccurrenceMapper.class,
                                                LongWritable.class,
                                                LongWritable.class,
                                                UserVectorToCooccurrenceReducer.class,
                                                LongWritable.class,
                                                Vector.class,
                                                SequenceFileOutputFormat.class);
    JobClient.runJob(toCooccurrenceConf);

    JobConf recommenderConf = prepareJobConf(cooccurrencePath,
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
    JobClient.runJob(recommenderConf);
  }

}
