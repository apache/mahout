package org.apache.mahout.classifier.bayes.common;
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

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.Iterator;


/**
 *  Can also be used as a local Combiner beacuse only two values should be there inside the values
 *
 **/

public class BayesTfIdfReducer extends MapReduceBase implements Reducer<Text, FloatWritable, Text, FloatWritable> {
  public void reduce(Text key, Iterator<FloatWritable> values, OutputCollector<Text, FloatWritable> output, Reporter reporter) throws IOException {
    //Key is label,word, value is the number of times we've seen this label word per local node.  Output is the same
    String token = key.toString();  
    if(token.startsWith("*vocabCount"))
    {
      float vocabCount = 0;
      while (values.hasNext()) {
        vocabCount += values.next().get();
      }
      System.out.println(token + "\t"+vocabCount);
      output.collect(key, new FloatWritable(vocabCount));
    }
    else
    {
      float idfTimes_D_ij = 1.0f;
      int numberofValues = 0;
      while (values.hasNext()) {
        idfTimes_D_ij *= values.next().get();
        numberofValues ++;
      }
      //System.out.println(token + "\t" + numberofValues + "\t"+idfTimes_D_ij);
      //if(numberofValues!=2) throw new IOException("Number of values should be exactly 2");
      
      output.collect(key, new FloatWritable(idfTimes_D_ij));
    }
  }
}
