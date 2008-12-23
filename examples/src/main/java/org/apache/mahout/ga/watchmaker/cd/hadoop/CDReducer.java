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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;

import java.io.IOException;
import java.util.Iterator;

/**
 * Hadoop Reducer, combines many evaluations (CDFitness) into a single one.
 */
public class CDReducer extends MapReduceBase implements
    Reducer<LongWritable, CDFitness, LongWritable, CDFitness> {

  @Override
  public void reduce(LongWritable key, Iterator<CDFitness> values,
      OutputCollector<LongWritable, CDFitness> output, Reporter reporter)
      throws IOException {
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;

    while (values.hasNext()) {
      CDFitness v = values.next();
      tp += v.getTp();
      fp += v.getFp();
      tn += v.getTn();
      fn += v.getFn();
    }

    output.collect(key, new CDFitness(tp, fp, tn, fn));
  }

}
