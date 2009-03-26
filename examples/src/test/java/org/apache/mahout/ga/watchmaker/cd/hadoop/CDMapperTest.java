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

package org.apache.mahout.ga.watchmaker.cd.hadoop;

import junit.framework.TestCase;
import org.apache.hadoop.io.LongWritable;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.DataLine;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.utils.DummyOutputCollector;
import org.easymock.classextension.EasyMock;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class CDMapperTest extends TestCase {

  DataLine dl;

  Rule rule;

  final CDFitness TP = new CDFitness(1, 0, 0, 0);

  final CDFitness FP = new CDFitness(0, 1, 0, 0);

  final CDFitness TN = new CDFitness(0, 0, 1, 0);

  final CDFitness FN = new CDFitness(0, 0, 0, 1);

  @Override
  protected void setUp() throws Exception {
    // we assume 2 classes 0 and 1
    // their are 4 tests
    // TP: dataline label 1, rule returns 1
    // FP: dataline label 0, rule returns 1
    // TN: dataline label 0, rule returns 0
    // FN: dataline label 1, rule returns 0

    dl = EasyMock.createMock(DataLine.class);
    EasyMock.expect(dl.getLabel()).andReturn(1);
    EasyMock.expect(dl.getLabel()).andReturn(0);
    EasyMock.expect(dl.getLabel()).andReturn(0);
    EasyMock.expect(dl.getLabel()).andReturn(1);

    rule = EasyMock.createMock(Rule.class);
    EasyMock.expect(rule.classify(dl)).andReturn(1);
    EasyMock.expect(rule.classify(dl)).andReturn(1);
    EasyMock.expect(rule.classify(dl)).andReturn(0);
    EasyMock.expect(rule.classify(dl)).andReturn(0);

    super.setUp();
  }

  public void testEvaluate() {
    // test the evaluation
    assertEquals(TP, CDMapper.evaluate(1, 1, 1));
    assertEquals(FP, CDMapper.evaluate(1, 1, 0));
    assertEquals(TN, CDMapper.evaluate(1, 0, 0));
    assertEquals(FN, CDMapper.evaluate(1, 0, 1));
  }

  public void testMap() throws Exception {
    EasyMock.replay(rule);
    EasyMock.replay(dl);

    // create and configure the mapper
    CDMapper mapper = new CDMapper();
    List<Rule> rules = Arrays.asList(rule, rule, rule, rule);
    mapper.configure(rules, 1);

    // test the mapper
    DummyOutputCollector<LongWritable, CDFitness> collector = new DummyOutputCollector<LongWritable, CDFitness>();
    mapper.map(new LongWritable(0), dl, collector);

    // check the evaluations
    Set<String> keys = collector.getKeys();
    assertEquals("Number of evaluations", rules.size(), keys.size());

    CDFitness[] expected = { TP, FP, TN, FN };
    for (String key : keys) {
      int index = Integer.parseInt(key);
      assertEquals("Values for key " + key, 1, collector.getValue(key).size());
      CDFitness eval = collector.getValue(key).get(0);

      assertEquals("Evaluation of the rule " + key, expected[index], eval);
    }

    EasyMock.verify(rule);
    EasyMock.verify(dl);
  }

}
