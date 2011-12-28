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
package org.apache.mahout.clustering.lda;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.IntegerDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleFunction;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import org.junit.Ignore;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

public final class TestMapReduce extends MahoutTestCase {

  private static final int NUM_TESTS = 10;
  private static final int NUM_TOPICS = 10;

  private Random random;

  /**
   * Generate random document vector
   * @param numWords int number of words in the vocabulary
   * @param numWords E[count] for each word
   */
  private RandomAccessSparseVector generateRandomDoc(int numWords, double sparsity) throws MathException {
    RandomAccessSparseVector v = new RandomAccessSparseVector(numWords,(int)(numWords * sparsity));
    IntegerDistribution dist = new PoissonDistributionImpl(sparsity);
    for (int i = 0; i < numWords; i++) {
      // random integer
      v.set(i,dist.inverseCumulativeProbability(random.nextDouble()) + 1);
    }
    return v;
  }

  private LDAState generateRandomState(int numWords, int numTopics) {
    double topicSmoothing = 50.0 / numTopics; // whatever
    Matrix m = new DenseMatrix(numTopics,numWords);
    double[] logTotals = new double[numTopics];
    for(int k = 0; k < numTopics; ++k) {
      double total = 0.0; // total number of pseudo counts we made
      for(int w = 0; w < numWords; ++w) {
        // A small amount of random noise, minimized by having a floor.
        double pseudocount = random.nextDouble() + 1.0E-10;
        total += pseudocount;
        m.setQuick(k,w,Math.log(pseudocount));
      }

      logTotals[k] = Math.log(total);
    }

    double ll = Double.NEGATIVE_INFINITY;
    return new LDAState(numTopics,numWords,topicSmoothing,m,logTotals,ll);
  }

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    random = RandomUtils.getRandom();
  }

  /**
   * Test the basic Mapper
   */
  @Test
  public void testMapper() throws Exception {
    LDAState state = generateRandomState(100,NUM_TOPICS);
    LDAWordTopicMapper mapper = new LDAWordTopicMapper();
    mapper.configure(state);
    for(int i = 0; i < NUM_TESTS; ++i) {
      RandomAccessSparseVector v = generateRandomDoc(100,0.3);
      int myNumWords = numNonZero(v);
      LDAWordTopicMapper.Context mock = EasyMock.createMock(LDAWordTopicMapper.Context.class);

      mock.write(EasyMock.isA(IntPairWritable.class), EasyMock.isA(DoubleWritable.class));
      EasyMock.expectLastCall().times(myNumWords * NUM_TOPICS + NUM_TOPICS + 1);
      EasyMock.replay(mock);
      VectorWritable vw = new VectorWritable(v);
      mapper.map(new Text("tstMapper"), vw, mock);
      EasyMock.verify(mock);
    }
  }

  @Test
  @Ignore("MAHOUT-399")
  public void testEndToEnd() throws Exception {
    int numGeneratingTopics = 5;
    int numTerms = 26;
    Matrix matrix = ClusteringTestUtils.randomStructuredModel(numGeneratingTopics, numTerms,
                                                              new DoubleFunction() {
                                                                @Override
                                                                public double apply(double d) {
                                                                  return 1.0 / Math.pow(d + 1, 3);
                                                                }
                                                              });

    int numDocs = 500;
    int numSamples = 10;
    int numTopicsPerDoc = 1;

    Matrix sampledCorpus = ClusteringTestUtils.sampledCorpus(matrix, RandomUtils.getRandom(),
        numDocs, numSamples, numTopicsPerDoc);

    Path sampleCorpusPath = getTestTempDirPath("corpus");
    MatrixUtils.write(sampleCorpusPath, new Configuration(), sampledCorpus);

    int numIterations = 10;
    List<Double> perplexities = Lists.newArrayList();
    int startTopic = numGeneratingTopics - 2;
    int numTestTopics = startTopic;
    double eta = 0.1;
    while(numTestTopics < numGeneratingTopics + 3) {
      LDADriver driver = new LDADriver();
      driver.setConf(new Configuration());
      Path outputPath = getTestTempDirPath("output" + numTestTopics);
      perplexities.add(driver.run(driver.getConf(), sampleCorpusPath, outputPath, numTestTopics,
          numTerms, eta, numIterations, false));
      numTestTopics++;
    }
    
    int bestTopic = -1;
    double lowestPerplexity = Double.MAX_VALUE;
    for(int t = 0; t < perplexities.size(); t++) {
      if(perplexities.get(t) < lowestPerplexity) {
        lowestPerplexity = perplexities.get(t);
        bestTopic = t + startTopic;
      }
    }
    assertEquals("The optimal number of topics is not that of the generating distribution",
        bestTopic, numGeneratingTopics);
    System.out.println("Perplexities: " + Joiner.on(", ").join(perplexities));
  }

  private static int numNonZero(Vector v) {
    int count = 0;
    for(Iterator<Vector.Element> iter = v.iterateNonZero();
        iter.hasNext();iter.next() ) {
      count++;
    }
    return count;
  }

}
