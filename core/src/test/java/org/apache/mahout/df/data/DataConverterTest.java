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

package org.apache.mahout.df.data;

import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;

public class DataConverterTest extends MahoutTestCase {

  private static final int nbAttributes = 10;
  
  private static final int nbInstances = 100;
  
  public void testConvert() throws Exception {
    Random rng = RandomUtils.getRandom();
    
    String descriptor = Utils.randomDescriptor(rng, nbAttributes);
    double[][] source = Utils.randomDoubles(rng, descriptor, nbInstances);
    String[] sData = Utils.double2String(source);
    Dataset dataset = DataLoader.generateDataset(descriptor, sData);
    Data data = DataLoader.loadData(dataset, sData);
    
    DataConverter converter = new DataConverter(dataset);
    
    for (int index = 0; index < data.size(); index++) {
      assertEquals(data.get(index), converter.convert(index, sData[index]));
    }
  }
}
