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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.IntegerTuple;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.fpm.pfpgrowth.convertors.ContextWriteOutputCollector;
import org.apache.mahout.fpm.pfpgrowth.convertors.integer.IntegerStringOutputConvertor;
import org.apache.mahout.fpm.pfpgrowth.convertors.integer.IntegerTupleIterator;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPTreeDepthCache;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * {@link ParallelFPGrowthReducer} takes each group of transactions and runs
 * Vanilla FPGrowth on it and outputs the the Top K frequent Patterns for each
 * group.
 * 
 */

public class ParallelFPGrowthReducer extends
    Reducer<LongWritable, IntegerTuple, Text, TopKStringPatterns> {

  private List<Pair<Integer, Long>> fList = new ArrayList<Pair<Integer, Long>>();
  
  private List<String> featureReverseMap = new ArrayList<String>();
  
  private Map<String, Integer> fMap = new HashMap<String, Integer>();

  private Map<Long, List<Integer>> groupFeatures = new HashMap<Long, List<Integer>>();

  private int maxHeapSize = 50;

  private int minSupport = 3;

  @Override
  public void reduce(LongWritable key, Iterable<IntegerTuple> values,
      Context context) throws IOException {
    FPGrowth<Integer> fpGrowth = new FPGrowth<Integer>();
    fpGrowth
        .generateTopKFrequentPatterns(
            new IntegerTupleIterator(values.iterator()),
            fList,
            minSupport,
            maxHeapSize,
            new HashSet<Integer>(groupFeatures.get(key.get())),
            new IntegerStringOutputConvertor(
                new ContextWriteOutputCollector<LongWritable, IntegerTuple, Text, TopKStringPatterns>(
                    context), featureReverseMap));
  }

  @Override
  public void setup(Context context) throws IOException, InterruptedException {

    super.setup(context);
    Parameters params = Parameters.fromString(context.getConfiguration().get(
        "pfp.parameters", ""));
    
    
    
    int i = 0;
    for(Pair<String, Long> e: PFPGrowth.deserializeList(params, "fList", context
        .getConfiguration()))
    {
      featureReverseMap.add(e.getFirst());
      fMap.put(e.getFirst(), i);
      fList.add(new Pair<Integer, Long>(i++, e.getSecond()));
    }
    
    Map<String, Long> gList = PFPGrowth.deserializeMap(params, "gList", context
        .getConfiguration());
    
    for (Entry<String, Long> entry : gList.entrySet()) {
      List<Integer> groupList = groupFeatures.get(entry.getValue());
      Integer itemInteger = fMap.get(entry.getKey());
      if (groupList != null)
        groupList.add(itemInteger);
      else {
        groupList = new ArrayList<Integer>();
        groupList.add(itemInteger);
        groupFeatures.put(entry.getValue(), groupList);
      }

    }
    maxHeapSize = Integer.valueOf(params.get("maxHeapSize", "50"));
    minSupport = Integer.valueOf(params.get("minSupport", "3"));
    FPTreeDepthCache.FirstLevelCacheSize = Integer.valueOf(params
        .get("treeCacheSize", Integer
            .toString(FPTreeDepthCache.FirstLevelCacheSize)));
  }
}
