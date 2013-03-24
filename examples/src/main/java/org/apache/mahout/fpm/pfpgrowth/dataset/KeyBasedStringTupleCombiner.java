/*
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
import java.util.Set;

import com.google.common.collect.Sets;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.StringTuple;

public class KeyBasedStringTupleCombiner extends Reducer<Text,StringTuple,Text,StringTuple> {
  
  @Override
  protected void reduce(Text key,
                        Iterable<StringTuple> values,
                        Context context) throws IOException, InterruptedException {
    Set<String> outputValues = Sets.newHashSet();
    for (StringTuple value : values) {
      outputValues.addAll(value.getEntries());
    }
    context.write(key, new StringTuple(outputValues));
  }
}
