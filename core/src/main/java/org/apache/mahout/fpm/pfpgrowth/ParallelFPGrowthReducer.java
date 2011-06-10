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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.google.common.collect.Lists;
import org.apache.commons.lang.mutable.MutableLong;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.fpm.pfpgrowth.convertors.ContextStatusUpdater;
import org.apache.mahout.fpm.pfpgrowth.convertors.ContextWriteOutputCollector;
import org.apache.mahout.fpm.pfpgrowth.convertors.integer.IntegerStringOutputConverter;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenLongObjectHashMap;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

/**
 *  takes each group of transactions and runs Vanilla FPGrowth on it and
 * outputs the the Top K frequent Patterns for each group.
 * 
 */
public class ParallelFPGrowthReducer extends Reducer<LongWritable,TransactionTree,Text,TopKStringPatterns> {
  
  private final List<String> featureReverseMap = Lists.newArrayList();
  
  private final OpenObjectIntHashMap<String> fMap = new OpenObjectIntHashMap<String>();
  
  private final OpenLongObjectHashMap<IntArrayList> groupFeatures = new OpenLongObjectHashMap<IntArrayList>();
  
  private int maxHeapSize = 50;
  
  private int minSupport = 3;
  
  @Override
  protected void reduce(LongWritable key, Iterable<TransactionTree> values, Context context) throws IOException {
    TransactionTree cTree = new TransactionTree();
    for (TransactionTree tr : values) {
      for (Pair<List<Integer>,Long> p : tr) {
        cTree.addPattern(p.getFirst(), p.getSecond());
      }
    }
    
    List<Pair<Integer,Long>> localFList = Lists.newArrayList();
    for (Entry<Integer,MutableLong> fItem : cTree.generateFList().entrySet()) {
      localFList.add(new Pair<Integer,Long>(fItem.getKey(), fItem.getValue().toLong()));
      
    }
    
    Collections.sort(localFList, new CountDescendingPairComparator<Integer,Long>());
    
    FPGrowth<Integer> fpGrowth = new FPGrowth<Integer>();
    fpGrowth.generateTopKFrequentPatterns(
        cTree.iterator(),
        localFList,
        minSupport,
        maxHeapSize,
        new HashSet<Integer>(groupFeatures.get(key.get()).toList()),
        new IntegerStringOutputConverter(
            new ContextWriteOutputCollector<LongWritable,TransactionTree,Text,TopKStringPatterns>(context),
            featureReverseMap),
        new ContextStatusUpdater<LongWritable,TransactionTree,Text,TopKStringPatterns>(context));
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    
    super.setup(context);
    Parameters params = new Parameters(context.getConfiguration().get(PFPGrowth.PFP_PARAMETERS, ""));
    
    int i = 0;
    for (Pair<String,Long> e : PFPGrowth.deserializeList(params, PFPGrowth.F_LIST, context.getConfiguration())) {
      featureReverseMap.add(e.getFirst());
      fMap.put(e.getFirst(), i++);
      
    }
    
    Map<String,Long> gList = PFPGrowth.deserializeMap(params, PFPGrowth.G_LIST, context.getConfiguration());
    
    for (Entry<String,Long> entry : gList.entrySet()) {
      IntArrayList groupList = groupFeatures.get(entry.getValue());
      Integer itemInteger = fMap.get(entry.getKey());
      if (groupList != null) {
        groupList.add(itemInteger);
      } else {
        groupList = new IntArrayList();
        groupList.add(itemInteger);
        groupFeatures.put(entry.getValue(), groupList);
      }
      
    }
    maxHeapSize = Integer.valueOf(params.get(PFPGrowth.MAX_HEAPSIZE, "50"));
    minSupport = Integer.valueOf(params.get(PFPGrowth.MIN_SUPPORT, "3"));
  }
}
