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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Pattern;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

/**
 *  maps each transaction to all unique items groups in the transaction.
 * mapper outputs the group id as key and the transaction as value
 * 
 */
public class TransactionSortingMapper extends Mapper<LongWritable,Text,LongWritable,TransactionTree> {
  
  private final OpenObjectIntHashMap<String> fMap = new OpenObjectIntHashMap<String>();
  
  private Pattern splitter;
  
  @Override
  protected void map(LongWritable offset, Text input, Context context) throws IOException,
                                                                      InterruptedException {
    
    String[] items = splitter.split(input.toString());
    Iterable<String> uniqueItems = new HashSet<String>(Arrays.asList(items));
    
    List<Integer> itemSet = Lists.newArrayList();
    for (String item : uniqueItems) { // remove items not in the fList
      if (fMap.containsKey(item) && item.trim().length() != 0) {
        itemSet.add(fMap.get(item));
      }
    }
    
    Collections.sort(itemSet);
    
    Integer[] prunedItems = itemSet.toArray(new Integer[itemSet.size()]);
    
    if (prunedItems.length > 0) {
      context.write(new LongWritable(prunedItems[0]), new TransactionTree(prunedItems, 1L));
    }
    
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Parameters params = new Parameters(context.getConfiguration().get(PFPGrowth.PFP_PARAMETERS, ""));
    
    int i = 0;
    for (Pair<String,Long> e : PFPGrowth.deserializeList(params, PFPGrowth.F_LIST, context.getConfiguration())) {
      fMap.put(e.getFirst(), i++);
    }
    
    splitter = Pattern.compile(params.get(PFPGrowth.SPLIT_PATTERN, PFPGrowth.SPLITTER.toString()));
    
  }
}
