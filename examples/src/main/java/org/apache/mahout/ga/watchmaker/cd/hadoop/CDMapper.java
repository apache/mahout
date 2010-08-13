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

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.DataLine;
import org.apache.mahout.ga.watchmaker.cd.DataSet;
import org.apache.mahout.ga.watchmaker.cd.Rule;

/**
 * Hadoop Mapper. Evaluate all the rules with the input data line.
 */
public class CDMapper extends Mapper<LongWritable, Text, LongWritable, CDFitness> {

  public static final String CLASSDISCOVERY_RULES = "mahout.ga.classdiscovery.rules";

  public static final String CLASSDISCOVERY_DATASET = "mahout.ga.classdiscovery.dataset";

  public static final String CLASSDISCOVERY_TARGET_LABEL = "mahout.ga.classdiscovery.target";

  List<Rule> rules;

  int target;

  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    DataLine dl = new DataLine(value.toString());

    for (int index = 0; index < rules.size(); index++) {
      CDFitness eval = evaluate(target, rules.get(index).classify(dl), dl.getLabel());
      context.write(new LongWritable(index), eval);
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    String rstr = conf.get(CLASSDISCOVERY_RULES);
    if (rstr == null) {
      throw new IllegalArgumentException("Job Parameter (" + CLASSDISCOVERY_RULES + ") not found!");
    }

    String datastr = conf.get(CLASSDISCOVERY_DATASET);
    if (datastr == null) {
      throw new IllegalArgumentException("Job Parameter (" + CLASSDISCOVERY_DATASET + ") not found!");
    }

    int target = conf.getInt(CLASSDISCOVERY_TARGET_LABEL, -1);
    if (target == -1) {
      throw new IllegalArgumentException("Job Parameter (" + CLASSDISCOVERY_TARGET_LABEL + ") not found!");
    }

    initializeDataSet(StringUtils.<DataSet> fromString(datastr));
    configure(StringUtils.<List<Rule>> fromString(rstr), target);
  }

  static void initializeDataSet(DataSet dataset) {
    if (dataset == null) {
      throw new IllegalArgumentException("bad 'dataset' configuration parameter");
    }
    DataSet.initialize(dataset);
  }

  void configure(List<Rule> rules, int target) {
    if (rules == null || rules.isEmpty()) {
      throw new IllegalArgumentException("bad 'rules' configuration parameter");
    }
    if (target < 0) {
      throw new IllegalArgumentException("bad 'target' configuration parameter");
    }

    this.rules = rules;
    this.target = target;

  }

  /**
   * Evaluate a given prediction.
   * 
   * @param target
   *          expected label
   * @param prediction
   * @param label
   *          actual label
   */
  public static CDFitness evaluate(int target, int prediction, int label) {
    boolean labelIsTarget = label == target;
    int tp = labelIsTarget && prediction == 1 ? 1 : 0;
    int fp = !labelIsTarget && prediction == 1 ? 1 : 0;
    int tn = !labelIsTarget && prediction == 0 ? 1 : 0;
    int fn = labelIsTarget && prediction == 0 ? 1 : 0;
    return new CDFitness(tp, fp, tn, fn);
  }
}
