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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.IOException;

/**
 * <p>This class configures and runs a {@link RecommenderMapper} using Hadoop.</p>
 *
 * <p>Command line arguments are:</p>
 * <ol>
 *  <li>Fully-qualified class name of {@link Recommender} to use to make recommendations.
 *   Note that it must have a no-arg constructor.</li>
 *  <li>Number of recommendations to compute per user</li>
 *  <li>Location of a text file containing user IDs for which recommendations should be computed,
 *   one per line</li>
 *  <li>Location of a data model file containing preference data, suitable for use with
 *   {@link org.apache.mahout.cf.taste.impl.model.file.FileDataModel}</li>
 *  <li>Output path where reducer output should go</li>
 *  <li>Number of mapper tasks to use</li>
 * </ol>
 *
 * <p>Example:</p>
 *
 * <p><code>org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender 10 path/to/users.txt
 *  path/to/data.csv path/to/reducerOutputDir 5</code></p>
 *
 * <p>TODO I am not a bit sure this works yet in a real distributed environment.</p>
 */
public final class RecommenderJob {
  private RecommenderJob() {
  }

  public static void main(String[] args) throws IOException {
    String recommendClassName = args[0];
    int recommendationsPerUser = Integer.parseInt(args[1]);
    String userIDFile = args[2];
    String dataModelFile = args[3];
    String outputPath = args[4];
    JobConf jobConf =
        buildJobConf(recommendClassName, recommendationsPerUser, userIDFile, dataModelFile, outputPath);
    JobClient.runJob(jobConf);
  }

  public static JobConf buildJobConf(String recommendClassName,
                                     int recommendationsPerUser,
                                     String userIDFile,
                                     String dataModelFile,
                                     String outputPath) throws IOException {

    Path userIDFilePath = new Path(userIDFile);
    Path outputPathPath = new Path(outputPath);

    JobConf jobConf = new JobConf(RecommenderJob.class);

    FileSystem fs = FileSystem.get(jobConf);
    if (fs.exists(outputPathPath)) {
      fs.delete(outputPathPath, true);
    }

    jobConf.set(RecommenderMapper.RECOMMENDER_CLASS_NAME, recommendClassName);
    jobConf.set(RecommenderMapper.RECOMMENDATIONS_PER_USER, String.valueOf(recommendationsPerUser));
    jobConf.set(RecommenderMapper.DATA_MODEL_FILE, dataModelFile);

    jobConf.setInputFormat(TextInputFormat.class);
    FileInputFormat.setInputPaths(jobConf, userIDFilePath);

    jobConf.setMapperClass(RecommenderMapper.class);
    jobConf.setMapOutputKeyClass(Text.class);
    jobConf.setMapOutputValueClass(RecommendedItemsWritable.class);

    jobConf.setReducerClass(IdentityReducer.class);
    jobConf.setOutputKeyClass(Text.class);
    jobConf.setOutputValueClass(RecommendedItemsWritable.class);

    jobConf.setOutputFormat(TextOutputFormat.class);
    FileOutputFormat.setOutputPath(jobConf, outputPathPath);

    return jobConf;
  }

}
