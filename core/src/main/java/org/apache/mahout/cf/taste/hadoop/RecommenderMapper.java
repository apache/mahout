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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

/**
 * <p>The {@link Mapper} which takes as input a file of user IDs (treated as Strings, note), and
 * for each unique user ID, computes recommendations with the configured {@link Recommender}.
 * The results are output as {@link RecommendedItemsWritable}.</p>
 *
 * <p>Note that there is no corresponding {@link org.apache.hadoop.mapred.Reducer}; this
 * implementation can only partially take advantage of the mapreduce paradigm and only
 * really leverages it for easy parallelization.</p>
 */
public final class RecommenderMapper
    extends MapReduceBase
    implements Mapper<LongWritable, Text, Text, RecommendedItemsWritable> {

  static final String RECOMMENDER_CLASS_NAME = "recommenderClassName";
  static final String RECOMMENDATIONS_PER_USER = "recommendadtionsPerUser";
  static final String DATA_MODEL_FILE = "dataModelFile";

  private Recommender recommender;
  private int recommendationsPerUser;

  @Override
  public void map(LongWritable key,
                  Text value,
                  OutputCollector<Text, RecommendedItemsWritable> output,
                  Reporter reporter) throws IOException {
    String userID = value.toString();
    List<RecommendedItem> recommendedItems;
    try {
      recommendedItems = recommender.recommend(userID, recommendationsPerUser);
    } catch (TasteException te) {
      throw new RuntimeException(te);
    }
    RecommendedItemsWritable writable = new RecommendedItemsWritable(recommendedItems);
    output.collect(new Text(userID), writable);
    reporter.incrCounter(ReducerMetrics.USERS_PROCESSED, 1L);
    reporter.incrCounter(ReducerMetrics.RECOMMENDATIONS_MADE, recommendedItems.size());
  }

  @Override
  public void configure(JobConf jobConf) {
    String dataModelFile = jobConf.get(DATA_MODEL_FILE);
    String recommenderClassName = jobConf.get(RECOMMENDER_CLASS_NAME);
    FileDataModel fileDataModel;
    try {
      FileSystem fs = FileSystem.get(jobConf);
      File tempDataFile = File.createTempFile("mahout-taste-hadoop", "txt");
      tempDataFile.deleteOnExit();
      fs.copyToLocalFile(new Path(dataModelFile), new Path(tempDataFile.getAbsolutePath()));
      fileDataModel = new FileDataModel(tempDataFile);
    } catch (IOException ioe) {
      throw new RuntimeException(ioe);
    }
    try {
      Class<? extends Recommender> recommenderClass = Class.forName(recommenderClassName).asSubclass(Recommender.class);
      Constructor<? extends Recommender> constructor = recommenderClass.getConstructor(DataModel.class);
      recommender = constructor.newInstance(fileDataModel);
    } catch (NoSuchMethodException nsme) {
      throw new RuntimeException(nsme);
    } catch (ClassNotFoundException cnfe) {
      throw new RuntimeException(cnfe);
    } catch (InstantiationException ie) {
      throw new RuntimeException(ie);
    } catch (IllegalAccessException iae) {
      throw new RuntimeException(iae);
    } catch (InvocationTargetException ite) {
      throw new RuntimeException(ite.getCause());
    }
    recommendationsPerUser = Integer.parseInt(jobConf.get(RECOMMENDATIONS_PER_USER));
  }

}
