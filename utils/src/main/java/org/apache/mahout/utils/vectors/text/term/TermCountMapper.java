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

package org.apache.mahout.utils.vectors.text.term;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.math.function.ObjectLongProcedure;
import org.apache.mahout.math.map.OpenObjectLongHashMap;

/**
 * TextVectorizer Term Count Mapper. Tokenizes a text document and outputs the count of the words
 * 
 */
public class TermCountMapper extends MapReduceBase implements Mapper<Text,StringTuple,Text,LongWritable> {
  @Override
  public void map(Text key,
                  StringTuple value,
                  final OutputCollector<Text,LongWritable> output,
                  final Reporter reporter) throws IOException {
    OpenObjectLongHashMap<String> wordCount = new OpenObjectLongHashMap<String>();
    for (String word : value.getEntries()) {
      if (wordCount.containsKey(word) == false) {
        wordCount.put(word, 1);
      } else {
        wordCount.put(word, wordCount.get(word) + 1);
      }
    }
    wordCount.forEachPair(new ObjectLongProcedure<String>() {
      @Override
      public boolean apply(String first, long second) {
        try {
          output.collect(new Text(first), new LongWritable(second));
        } catch (IOException e) {
          reporter.incrCounter("Exception", "Output IO Exception", 1);
        }
        return true;
      }
    });
  }
}
