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

package org.apache.mahout.fpm.pfpgrowth.dataset;

import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;

public class KeyBasedStringTupleReducer extends Reducer<Text,StringTuple,Text,Text> {
  
  private int maxTransactionLength = 100;
  
  @Override
  protected void reduce(Text key, Iterable<StringTuple> values, Context context)
    throws IOException, InterruptedException {
    Collection<String> items = new HashSet<>();
    
    for (StringTuple value : values) {
      for (String field : value.getEntries()) {
        items.add(field);
      }
    }
    if (items.size() > 1) {
      int i = 0;
      StringBuilder sb = new StringBuilder();
      String sep = "";
      for (String field : items) {
        if (i % maxTransactionLength == 0) {
          if (i != 0) {
            context.write(null, new Text(sb.toString()));
          }
          sb.replace(0, sb.length(), "");
          sep = "";
        }
        
        sb.append(sep).append(field);
        sep = "\t";
        
        i++;
        
      }
      if (sb.length() > 0) {
        context.write(null, new Text(sb.toString()));
      }
    }
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Parameters params = new Parameters(context.getConfiguration().get("job.parameters", ""));
    maxTransactionLength = Integer.valueOf(params.get("maxTransactionLength", "100"));
  }
}
