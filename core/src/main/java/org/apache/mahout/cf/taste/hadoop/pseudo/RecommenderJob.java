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

package org.apache.mahout.cf.taste.hadoop.pseudo;

import org.apache.commons.cli2.Option;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.AbstractJob;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.IOException;
import java.util.Map;

/**
 * <p>This job runs a "pseudo-distributed" recommendation process on Hadoop.
 * It merely runs many {@link Recommender} instances on Hadoop, where each instance
 * is a normal non-distributed implementation.</p>
 *
 * <p>This class configures and runs a {@link RecommenderMapper} using Hadoop.</p>
 *
 * <p>Command line arguments are:</p>
 *
 * <ol>
 *  <li>recommenderClassName: Fully-qualified class name of {@link Recommender} to use to make
 *   recommendations. Note that it must have a constructor which takes a
 *   {@link org.apache.mahout.cf.taste.model.DataModel} argument.</li>
 *  <li>numRecommendations: Number of recommendations to compute per user</li>
 *  <li>input: Directory containing a text file containing user IDs
 *   for which recommendations should be computed, one per line</li>
 *  <li>dataModelFile: Location of a data model file containing preference data,
 *   suitable for use with {@link org.apache.mahout.cf.taste.impl.model.file.FileDataModel}</li>
 *  <li>output: output path where recommender output should go</li>
 *  <li>jarFile: JAR file containing implementation code</li>
 * </ol>
 *
 * <p>For example, to get started trying this out, set up Hadoop in a
 * pseudo-distributed manner: http://hadoop.apache.org/common/docs/current/quickstart.html
 * You can stop at the point where it instructs you to copy files into HDFS.</p>
 *
 * <p>Assume your preference data file is <code>input.csv</code>. You will also need to create a file
 * containing all user IDs to write recommendations for, as something like <code>users.txt</code>.
 * Place this input on HDFS like so:</p>
 *
 * {@code
 * hadoop fs -put input.csv input/input.csv
 * hadoop fs -put users.txt input/users.txt
 * }
 *
 * <p>Build Mahout code with <code>mvn package</code> in the core/ directory. Locate
 * <code>target/mahout-core-X.Y-SNAPSHOT.job</code>. This is a JAR file; copy it out
 * to a convenient location and name it <code>recommender.jar</code>.</p>
 *
 * <p>Now add your own custom recommender code and dependencies. Your IDE produced compiled .class
 * files somewhere and they need to be packaged up as well:</p>
 *
 * {@code
 * jar uf recommender.jar -C (your classes directory) .
 * }
 *
 * <p>And launch:</p>
 *
 * {@code
 * hadoop jar recommender.jar org.apache.mahout.cf.taste.hadoop.RecommenderJob \
 *   --recommenderClassName your.project.Recommender \
 *   --numRecommendations 10 --input input/users.txt --dataModelFile input/input.csv \
 *   --output output --jarFile recommender.jar
 * }
 */
public final class RecommenderJob extends AbstractJob {

  @Override
  public int run(String[] args) throws IOException {

    Option recommendClassOpt = buildOption("recommenderClassName", "r", "Name of recommender class to instantiate", true);
    Option numReccomendationsOpt = buildOption("numRecommendations", "n", "Number of recommendations per user", true);
    Option dataModelFileOpt = buildOption("dataModelFile", "m", "File containing preference data", true);

    Map<String,Object> parsedArgs = parseArguments(args, recommendClassOpt, numReccomendationsOpt, dataModelFileOpt);
    String userIDFile = parsedArgs.get("--input").toString();
    String outputPath = parsedArgs.get("--output").toString();
    String jarFile = parsedArgs.get("--jarFile").toString();

    String recommendClassName = parsedArgs.get("--recommenderClassName").toString();
    int recommendationsPerUser = Integer.parseInt((String) parsedArgs.get("--numRecommendations"));
    String dataModelFile = parsedArgs.get("--dataModelFile").toString();

    JobConf jobConf = prepareJobConf(userIDFile,
                                     outputPath,
                                     jarFile,
                                     TextInputFormat.class,
                                     RecommenderMapper.class,
                                     LongWritable.class,
                                     RecommendedItemsWritable.class,
                                     IdentityReducer.class,
                                     LongWritable.class,
                                     RecommendedItemsWritable.class,
                                     TextOutputFormat.class);

    jobConf.set(RecommenderMapper.RECOMMENDER_CLASS_NAME, recommendClassName);
    jobConf.setInt(RecommenderMapper.RECOMMENDATIONS_PER_USER, recommendationsPerUser);
    jobConf.set(RecommenderMapper.DATA_MODEL_FILE, dataModelFile);

    JobClient.runJob(jobConf);
    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RecommenderJob(), args);
  }

}
