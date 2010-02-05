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
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang.mutable.MutableLong;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.StringTuple;

/**
 * TextVectorizer Term Count Mapper. Tokenizes a text document and outputs the
 * count of the words
 * 
 */
public class TermCountMapper extends MapReduceBase implements
    Mapper<Text,StringTuple,Text,LongWritable> {
  @Override
  public void map(Text key,
                  StringTuple value,
                  OutputCollector<Text,LongWritable> output,
                  Reporter reporter) throws IOException {
    
    Map<String,MutableLong> wordCount = new HashMap<String,MutableLong>();
    for (String word : value.getEntries()) {
      if (wordCount.containsKey(word) == false) {
        wordCount.put(word, new MutableLong(0));
      }
      wordCount.get(word).increment();
    }
    
    for (Entry<String,MutableLong> entry : wordCount.entrySet()) {
      output.collect(new Text(entry.getKey()), new LongWritable(entry
          .getValue().longValue()));
    }
  }
}
