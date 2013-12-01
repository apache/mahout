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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Run ensemble learning via loading the {@link ModelTrainer} with two {@link TopicModel} instances:
 * one from the previous iteration, the other empty.  Inference is done on the first, and the
 * learning updates are stored in the second, and only emitted at cleanup().
 * <p/>
 * In terms of obvious performance improvements still available, the memory footprint in this
 * Mapper could be dropped by half if we accumulated model updates onto the model we're using
 * for inference, which might also speed up convergence, as we'd be able to take advantage of
 * learning <em>during</em> iteration, not just after each one is done.  Most likely we don't
 * really need to accumulate double values in the model either, floats would most likely be
 * sufficient.  Between these two, we could squeeze another factor of 4 in memory efficiency.
 * <p/>
 * In terms of CPU, we're re-learning the p(topic|doc) distribution on every iteration, starting
 * from scratch.  This is usually only 10 fixed-point iterations per doc, but that's 10x more than
 * only 1.  To avoid having to do this, we would need to do a map-side join of the unchanging
 * corpus with the continually-improving p(topic|doc) matrix, and then emit multiple outputs
 * from the mappers to make sure we can do the reduce model averaging as well.  Tricky, but
 * possibly worth it.
 * <p/>
 * {@link ModelTrainer} already takes advantage (in maybe the not-nice way) of multi-core
 * availability by doing multithreaded learning, see that class for details.
 */
public class CachingCVB0Mapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

  private static final Logger log = LoggerFactory.getLogger(CachingCVB0Mapper.class);

  private ModelTrainer modelTrainer;
  private TopicModel readModel;
  private TopicModel writeModel;
  private int maxIters;
  private int numTopics;

  protected ModelTrainer getModelTrainer() {
    return modelTrainer;
  }

  protected int getMaxIters() {
    return maxIters;
  }

  protected int getNumTopics() {
    return numTopics;
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    log.info("Retrieving configuration");
    Configuration conf = context.getConfiguration();
    float eta = conf.getFloat(CVB0Driver.TERM_TOPIC_SMOOTHING, Float.NaN);
    float alpha = conf.getFloat(CVB0Driver.DOC_TOPIC_SMOOTHING, Float.NaN);
    long seed = conf.getLong(CVB0Driver.RANDOM_SEED, 1234L);
    numTopics = conf.getInt(CVB0Driver.NUM_TOPICS, -1);
    int numTerms = conf.getInt(CVB0Driver.NUM_TERMS, -1);
    int numUpdateThreads = conf.getInt(CVB0Driver.NUM_UPDATE_THREADS, 1);
    int numTrainThreads = conf.getInt(CVB0Driver.NUM_TRAIN_THREADS, 4);
    maxIters = conf.getInt(CVB0Driver.MAX_ITERATIONS_PER_DOC, 10);
    float modelWeight = conf.getFloat(CVB0Driver.MODEL_WEIGHT, 1.0f);

    log.info("Initializing read model");
    Path[] modelPaths = CVB0Driver.getModelPaths(conf);
    if (modelPaths != null && modelPaths.length > 0) {
      readModel = new TopicModel(conf, eta, alpha, null, numUpdateThreads, modelWeight, modelPaths);
    } else {
      log.info("No model files found");
      readModel = new TopicModel(numTopics, numTerms, eta, alpha, RandomUtils.getRandom(seed), null,
          numTrainThreads, modelWeight);
    }

    log.info("Initializing write model");
    writeModel = modelWeight == 1
        ? new TopicModel(numTopics, numTerms, eta, alpha, null, numUpdateThreads)
        : readModel;

    log.info("Initializing model trainer");
    modelTrainer = new ModelTrainer(readModel, writeModel, numTrainThreads, numTopics, numTerms);
    modelTrainer.start();
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
      throws IOException, InterruptedException {
    /* where to get docTopics? */
    Vector topicVector = new DenseVector(numTopics).assign(1.0 / numTopics);
    modelTrainer.train(document.get(), topicVector, true, maxIters);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    log.info("Stopping model trainer");
    modelTrainer.stop();

    log.info("Writing model");
    TopicModel readFrom = modelTrainer.getReadModel();
    for (MatrixSlice topic : readFrom) {
      context.write(new IntWritable(topic.index()), new VectorWritable(topic.vector()));
    }
    readModel.stop();
    writeModel.stop();
  }
}
