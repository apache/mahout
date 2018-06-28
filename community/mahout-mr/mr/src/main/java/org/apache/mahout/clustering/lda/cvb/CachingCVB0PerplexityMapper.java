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
package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.MemoryUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

public class CachingCVB0PerplexityMapper extends
    Mapper<IntWritable, VectorWritable, DoubleWritable, DoubleWritable> {
  /**
   * Hadoop counters for {@link CachingCVB0PerplexityMapper}, to aid in debugging.
   */
  public enum Counters {
    SAMPLED_DOCUMENTS
  }

  private static final Logger log = LoggerFactory.getLogger(CachingCVB0PerplexityMapper.class);

  private ModelTrainer modelTrainer;
  private TopicModel readModel;
  private int maxIters;
  private int numTopics;
  private float testFraction;
  private Random random;
  private Vector topicVector;
  private final DoubleWritable outKey = new DoubleWritable();
  private final DoubleWritable outValue = new DoubleWritable();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    MemoryUtil.startMemoryLogger(5000);

    log.info("Retrieving configuration");
    Configuration conf = context.getConfiguration();
    float eta = conf.getFloat(CVB0Driver.TERM_TOPIC_SMOOTHING, Float.NaN);
    float alpha = conf.getFloat(CVB0Driver.DOC_TOPIC_SMOOTHING, Float.NaN);
    long seed = conf.getLong(CVB0Driver.RANDOM_SEED, 1234L);
    random = RandomUtils.getRandom(seed);
    numTopics = conf.getInt(CVB0Driver.NUM_TOPICS, -1);
    int numTerms = conf.getInt(CVB0Driver.NUM_TERMS, -1);
    int numUpdateThreads = conf.getInt(CVB0Driver.NUM_UPDATE_THREADS, 1);
    int numTrainThreads = conf.getInt(CVB0Driver.NUM_TRAIN_THREADS, 4);
    maxIters = conf.getInt(CVB0Driver.MAX_ITERATIONS_PER_DOC, 10);
    float modelWeight = conf.getFloat(CVB0Driver.MODEL_WEIGHT, 1.0f);
    testFraction = conf.getFloat(CVB0Driver.TEST_SET_FRACTION, 0.1f);

    log.info("Initializing read model");
    Path[] modelPaths = CVB0Driver.getModelPaths(conf);
    if (modelPaths != null && modelPaths.length > 0) {
      readModel = new TopicModel(conf, eta, alpha, null, numUpdateThreads, modelWeight, modelPaths);
    } else {
      log.info("No model files found");
      readModel = new TopicModel(numTopics, numTerms, eta, alpha, RandomUtils.getRandom(seed), null,
          numTrainThreads, modelWeight);
    }

    log.info("Initializing model trainer");
    modelTrainer = new ModelTrainer(readModel, null, numTrainThreads, numTopics, numTerms);

    log.info("Initializing topic vector");
    topicVector = new DenseVector(new double[numTopics]);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    readModel.stop();
    MemoryUtil.stopMemoryLogger();
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
    throws IOException, InterruptedException {
    if (testFraction < 1.0f && random.nextFloat() >= testFraction) {
      return;
    }
    context.getCounter(Counters.SAMPLED_DOCUMENTS).increment(1);
    outKey.set(document.get().norm(1));
    outValue.set(modelTrainer.calculatePerplexity(document.get(), topicVector.assign(1.0 / numTopics), maxIters));
    context.write(outKey, outValue);
  }
}
