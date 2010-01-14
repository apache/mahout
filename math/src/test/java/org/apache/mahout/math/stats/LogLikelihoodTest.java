package org.apache.mahout.math.stats;
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

import org.junit.Assert;
import org.junit.Test;

/**
 *
 *
 **/
public class LogLikelihoodTest extends Assert{
  /*
  *> entropy(c(1,1))
[1] 1.386294
llr(matrix(c(1,0,0,1), nrow=2))
[1] 2.772589
llr(matrix(c(10,0,0,10), nrow=2))
[1] 27.72589
llr(matrix(c(5,1995,0,100000), nrow=2))
[1] 39.33052
llr(matrix(c(1000,1995,1000,100000), nrow=2))
[1] 4730.737
llr(matrix(c(1000,1000,1000,100000), nrow=2))
[1] 5734.343
llr(matrix(c(1000,1000,1000,99000), nrow=2))
[1] 5714.932
*
   */
  @Test
  public void testEntropy() throws Exception {

    assertEquals(LogLikelihood.entropy(1, 1), 1.386294, 0.0001);
    //TODO: more tests here
    try {
      LogLikelihood.entropy(-1, -1);//exception
      assertFalse(true);
    } catch (IllegalArgumentException e) {
      
    }
  }

  @Test
  public void testLogLikelihood() throws Exception {
    //TODO: check the epsilons
    assertEquals(LogLikelihood.logLikelihoodRatio(1,0,0,1), 2.772589, 0.0001);
    assertEquals(LogLikelihood.logLikelihoodRatio(10,0,0,10), 27.72589, 0.0001);
    assertEquals(LogLikelihood.logLikelihoodRatio(5,1995,0,100000), 39.33052, 0.0001);
    assertEquals(LogLikelihood.logLikelihoodRatio(1000,1995, 1000, 100000), 4730.737, 0.001);
    assertEquals(LogLikelihood.logLikelihoodRatio(1000,1000,1000, 100000), 5734.343, 0.001);
    assertEquals(LogLikelihood.logLikelihoodRatio(1000,1000,1000, 99000), 5714.932, 0.001);
  }

  @Test
  public void testRootLogLikelihood() throws Exception {
    // positive where k11 is bigger than expected.
    assertTrue(LogLikelihood.rootLogLikelihoodRatio(904, 21060, 1144, 283012) > 0.0);
    
    // negative because k11 is lower than expected
    assertTrue(LogLikelihood.rootLogLikelihoodRatio(36, 21928, 60280, 623876) < 0.0);
  }
}
