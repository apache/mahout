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
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

/**
 * TextVectorizer Document Frequency Count Mapper. Outputs 1 for each feature
 * 
 */
public class TermDocumentCountMapper extends MapReduceBase implements
    Mapper<WritableComparable<?>,VectorWritable,IntWritable,LongWritable> {
  
  private static final LongWritable ONE = new LongWritable(1);
  private static final IntWritable TOTAL_COUNT = new IntWritable(-1);
  
  @Override
  public void map(WritableComparable<?> key,
                  VectorWritable value,
                  OutputCollector<IntWritable,LongWritable> output,
                  Reporter reporter) throws IOException {
    Vector vector = value.get();
    Iterator<Element> it = vector.iterateNonZero();
    
    while (it.hasNext()) {
      Element e = it.next();
      output.collect(new IntWritable(e.index()), ONE);
    }
    output.collect(TOTAL_COUNT, ONE);
  }
}
