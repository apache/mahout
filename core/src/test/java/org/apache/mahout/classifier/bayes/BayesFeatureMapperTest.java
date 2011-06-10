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

package org.apache.mahout.classifier.bayes;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesFeatureMapper;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.StringTuple;
import org.junit.Test;

public final class BayesFeatureMapperTest extends MahoutTestCase {

  @Test
  public void test() throws Exception {
    BayesFeatureMapper mapper = new BayesFeatureMapper();
    JobConf conf = new JobConf();
    conf.set("io.serializations",
      "org.apache.hadoop.io.serializer.JavaSerialization,"
          + "org.apache.hadoop.io.serializer.WritableSerialization");
    BayesParameters bayesParams = new BayesParameters();
    bayesParams.setGramSize(3);
    conf.set("bayes.parameters", bayesParams.toString());
    mapper.configure(conf);
    
    DummyOutputCollector<StringTuple,DoubleWritable> output = new DummyOutputCollector<StringTuple,DoubleWritable>();
    mapper.map(new Text("foo"), new Text("big brown shoe"), output,
      Reporter.NULL);
    Map<StringTuple, List<DoubleWritable>> outMap = output.getData();
    assertNotNull("outMap is null and it shouldn't be", outMap);
    // TODO: How about not such a lame test here?
    for (Entry<StringTuple, List<DoubleWritable>> entry : outMap.entrySet()) {
      assertTrue("entry.getKey() Size: " + entry.getKey().length()
                 + " is not greater than: 0", entry.getKey().length() > 0);
      assertEquals("entry.getValue() Size: " + entry.getValue().size()
                   + " is not: 1", 1, entry.getValue().size());
      assertTrue("value is not valie", entry.getValue().get(0).get() > 0);
    }
    
  }
  
}
