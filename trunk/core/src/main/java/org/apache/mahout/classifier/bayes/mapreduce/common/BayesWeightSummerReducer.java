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

package org.apache.mahout.classifier.bayes.mapreduce.common;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Can also be used as a local Combiner */
public class BayesWeightSummerReducer extends MapReduceBase implements
    Reducer<StringTuple,DoubleWritable,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(BayesWeightSummerReducer.class);
  
  private HTable table;
  
  private boolean useHbase;
  
  @Override
  public void reduce(StringTuple key,
                     Iterator<DoubleWritable> values,
                     OutputCollector<StringTuple,DoubleWritable> output,
                     Reporter reporter) throws IOException {
    // Key is label,word, value is the tfidf of the feature of times we've seen
    // this label word per local node. Output is the same
    
    double sum = 0.0;
    while (values.hasNext()) {
      reporter.setStatus("Weight Summer Reducer: " + key);
      sum += values.next().get();
    }
    reporter.setStatus("Bayes Weight Summer Reducer: " + key + " => " + sum);
    // char firstChar = key.toString().charAt(0);
    if (useHbase) {
      if (key.stringAt(0).equals(BayesConstants.FEATURE_SUM)) { // sum of weight
        // for all
        // labels for a
        // feature
        // Sigma_j
        String feature = key.stringAt(1);
        
        Put bu = new Put(Bytes.toBytes(feature));
        bu.add(Bytes.toBytes(BayesConstants.HBASE_COLUMN_FAMILY), Bytes.toBytes(BayesConstants.FEATURE_SUM),
          Bytes.toBytes(sum));
        table.put(bu);
        
      } else if (key.stringAt(0).equals(BayesConstants.LABEL_SUM)) {
        String label = key.stringAt(1);
        Put bu = new Put(Bytes.toBytes(BayesConstants.LABEL_SUM));
        bu.add(Bytes.toBytes(BayesConstants.HBASE_COLUMN_FAMILY), Bytes.toBytes(label), Bytes.toBytes(sum));
        table.put(bu);
      } else if (key.stringAt(0).equals(BayesConstants.TOTAL_SUM)) {
        Put bu = new Put(Bytes.toBytes(BayesConstants.HBASE_COUNTS_ROW));
        bu.add(Bytes.toBytes(BayesConstants.HBASE_COLUMN_FAMILY), Bytes.toBytes(BayesConstants.TOTAL_SUM),
          Bytes.toBytes(sum));
        table.put(bu);
      }
    }
    
    output.collect(key, new DoubleWritable(sum));
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      Parameters params = Parameters.fromString(job.get("bayes.parameters", ""));
      if (params.get("dataSource").equals("hbase")) {
        useHbase = true;
      } else {
        return;
      }
      
      HBaseConfiguration hBconf = new HBaseConfiguration(job);
      table = new HTable(hBconf, job.get("output.table"));
    } catch (IOException e) {
      log.error("Unexpected error during configuration", e);
    }
    
  }
  
  @Override
  public void close() throws IOException {
    if (useHbase) {
      table.close();
    }
    super.close();
  }
}
