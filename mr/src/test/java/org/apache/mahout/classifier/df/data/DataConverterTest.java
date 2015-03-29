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

package org.apache.mahout.classifier.df.data;

import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

public final class DataConverterTest extends MahoutTestCase {

  private static final int ATTRIBUTE_COUNT = 10;
  
  private static final int INSTANCE_COUNT = 100;

  @Test
  public void testConvert() throws Exception {
    Random rng = RandomUtils.getRandom();
    
    String descriptor = Utils.randomDescriptor(rng, ATTRIBUTE_COUNT);
    double[][] source = Utils.randomDoubles(rng, descriptor, false, INSTANCE_COUNT);
    String[] sData = Utils.double2String(source);
    Dataset dataset = DataLoader.generateDataset(descriptor, false, sData);
    Data data = DataLoader.loadData(dataset, sData);
    
    DataConverter converter = new DataConverter(dataset);
    
    for (int index = 0; index < data.size(); index++) {
      assertEquals(data.get(index), converter.convert(sData[index]));
    }

    // regression
    source = Utils.randomDoubles(rng, descriptor, true, INSTANCE_COUNT);
    sData = Utils.double2String(source);
    dataset = DataLoader.generateDataset(descriptor, true, sData);
    data = DataLoader.loadData(dataset, sData);
    
    converter = new DataConverter(dataset);
    
    for (int index = 0; index < data.size(); index++) {
      assertEquals(data.get(index), converter.convert(sData[index]));
    }
  }
}
