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

package org.apache.mahout.classifier.df.mapreduce.partial;

import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

public final class TreeIDTest extends MahoutTestCase {

  @Test
  public void testTreeID() {
    Random rng = RandomUtils.getRandom();
    
    for (int nloop = 0; nloop < 1000000; nloop++) {
      int partition = Math.abs(rng.nextInt());
      int treeId = rng.nextInt(TreeID.MAX_TREEID);
      
      TreeID t1 = new TreeID(partition, treeId);
      
      assertEquals(partition, t1.partition());
      assertEquals(treeId, t1.treeId());
      
      TreeID t2 = new TreeID();
      t2.set(partition, treeId);

      assertEquals(partition, t2.partition());
      assertEquals(treeId, t2.treeId());
    }
  }
}
