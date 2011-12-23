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

package org.apache.mahout.classifier.df.builder;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Utils;
import org.junit.Test;

import java.util.Random;

public final class InfiniteRecursionTest extends MahoutTestCase {

  private static final double[][] dData = {
      { 0.25, 0.0, 0.0, 5.143998668220409E-4, 0.019847102289905324, 3.5216524641879855E-4, 0.0, 0.6225857142857143, 4 },
      { 0.25, 0.0, 0.0, 0.0010504411519893459, 0.005462138323171171, 0.0026130744829756746, 0.0, 0.4964857142857143, 3 },
      { 0.25, 0.0, 0.0, 0.0010504411519893459, 0.005462138323171171, 0.0026130744829756746, 0.0, 0.4964857142857143, 4 },
      { 0.25, 0.0, 0.0, 5.143998668220409E-4, 0.019847102289905324, 3.5216524641879855E-4, 0.0, 0.6225857142857143, 3 }
  };

  /**
   * make sure DecisionTreeBuilder.build() does not throw a StackOverflowException
   */
  @Test
  public void testBuild() throws Exception {
    Random rng = RandomUtils.getRandom();

    String[] source = Utils.double2String(dData);
    String descriptor = "N N N N N N N N L";

    Dataset dataset = DataLoader.generateDataset(descriptor, false, source);
    Data data = DataLoader.loadData(dataset, source);
    TreeBuilder builder = new DecisionTreeBuilder();
    builder.build(rng, data);

    // regression
    dataset = DataLoader.generateDataset(descriptor, true, source);
    data = DataLoader.loadData(dataset, source);
    builder = new DecisionTreeBuilder();
    builder.build(rng, data);
  }
}
