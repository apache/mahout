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

package org.apache.mahout.fpm.pfpgrowth;

import java.io.IOException;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;

/**
 * 
 *  outputs the pattern for each item in the pattern, so that reducer can group them
 * and select the top K frequent patterns
 * 
 */
public class AggregatorMapper extends Mapper<Text,TopKStringPatterns,Text,TopKStringPatterns> {
  
  @Override
  protected void map(Text key, TopKStringPatterns values, Context context) throws IOException,
                                                                          InterruptedException {
    for (Pair<List<String>,Long> pattern : values.getPatterns()) {
      for (String item : pattern.getFirst()) {
        List<Pair<List<String>,Long>> patternSingularList = Lists.newArrayList();
        patternSingularList.add(pattern);
        context.setStatus("Aggregator Mapper:Grouping Patterns for " + item);
        context.write(new Text(item), new TopKStringPatterns(patternSingularList));
      }
    }
    
  }
}
