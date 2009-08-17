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

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import junit.framework.TestCase;


import org.apache.commons.math.distribution.PoissonDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.matrix.DenseMatrix;
import org.apache.mahout.matrix.Matrix;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DummyOutputCollector;

import static org.easymock.classextension.EasyMock.*;

public class TestMapReduce extends TestCase {


  private Random random;

  /**
   * Generate random document vector
   * @param numWords int number of words in the vocabulary
   * @param numWords E[count] for each word
   */
  private SparseVector generateRandomDoc(int numWords, double sparsity) {
    SparseVector v = new SparseVector(numWords,(int)(numWords * sparsity));
    try {
      PoissonDistribution dist = new PoissonDistributionImpl(sparsity);
      for (int i = 0; i < numWords; i++) {
        // random integer
        v.set(i,dist.inverseCumulativeProbability(random.nextDouble()) + 1);
      }
    } catch(Exception e) {
      e.printStackTrace();
      fail("Caught " + e.toString());
    }
    return v;
  }

  private LDAState generateRandomState(int numWords, int numTopics) {
    double topicSmoothing = 50.0 / numTopics; // whatever
    Matrix m = new DenseMatrix(numTopics,numWords);
    double[] logTotals = new double[numTopics];
    double ll = Double.NEGATIVE_INFINITY;
    for(int k = 0; k < numTopics; ++k) {
      double total = 0.0; // total number of pseudo counts we made
      for(int w = 0; w < numWords; ++w) {
        // A small amount of random noise, minimized by having a floor.
        double pseudocount = random.nextDouble() + 1E-10;
        total += pseudocount;
        m.setQuick(k,w,Math.log(pseudocount));
      }

      logTotals[k] = Math.log(total);
    }

    return new LDAState(numTopics,numWords,topicSmoothing,m,logTotals,ll);
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    File f = new File("input");
    random = new Random();
    f.mkdir();
  }

  private static int NUM_TESTS = 10;
  private static int NUM_TOPICS = 10;

  /**
   * Test the basic Mapper
   * 
   * @throws Exception
   */
  public void testMapper() throws Exception {
    LDAState state = generateRandomState(100,NUM_TOPICS);
    LDAMapper mapper = new LDAMapper();
    mapper.configure(state);

    for(int i = 0; i < NUM_TESTS; ++i) {
      SparseVector v = generateRandomDoc(100,0.3);
      int myNumWords = numNonZero(v);
      LDAMapper.Context mock = createMock(LDAMapper.Context.class);

      mock.write(isA(IntPairWritable.class),isA(DoubleWritable.class));
      expectLastCall().times(myNumWords * NUM_TOPICS + NUM_TOPICS + 1);
      replay(mock);

      mapper.map(new Text("tstMapper"), v, mock);
      verify(mock);
    }
  }

  private int numNonZero(Vector v) {
    int count = 0;
    for(Iterator<Vector.Element> iter = v.iterateNonZero();
        iter.hasNext();iter.next() ) {
      count++;
    }
    return count;
  }

}
