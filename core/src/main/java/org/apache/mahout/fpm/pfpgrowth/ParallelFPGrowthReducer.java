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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPTreeDepthCache;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenLongObjectHashMap;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

/**
 *  takes each group of transactions and runs Vanilla FPGrowth on it and
 * outputs the the Top K frequent Patterns for each group.
 * 
 */

public class ParallelFPGrowthReducer extends Reducer<LongWritable,TransactionTree,Text,TopKStringPatterns> {
  
  private final List<Pair<Integer,Long>> fList = new ArrayList<Pair<Integer,Long>>();
  
  private final List<String> featureReverseMap = new ArrayList<String>();
  
  private final OpenObjectIntHashMap<String> fMap = new OpenObjectIntHashMap<String>();
  
  private final List<String> fRMap = new ArrayList<String>();
  
  private final OpenLongObjectHashMap<IntArrayList> groupFeatures = new OpenLongObjectHashMap<IntArrayList>();
  
  private int maxHeapSize = 50;
  
  private int minSupport = 3;
  
  @Override
  protected void reduce(LongWritable key, Iterable<TransactionTree> values, Context context) throws IOException {
    TransactionTree cTree = new TransactionTree();
    int nodes = 0;
    for (TransactionTree tr : values) {
      Iterator<Pair<List<Integer>,Long>> it = tr.getIterator();
      while (it.hasNext()) {
        Pair<List<Integer>,Long> p = it.next();
        nodes += cTree.addPattern(p.getFirst(), p.getSecond());
      }
    }
    
    List<Pair<Integer,Long>> localFList = new ArrayList<Pair<Integer,Long>>();
    for (Entry<Integer,MutableLong> fItem : cTree.generateFList().entrySet()) {
      localFList.add(new Pair<Integer,Long>(fItem.getKey(), fItem.getValue().toLong()));
      
    }
    
    Collections.sort(localFList, new Comparator<Pair<Integer,Long>>() {
      
      @Override
      public int compare(Pair<Integer,Long> o1, Pair<Integer,Long> o2) {
        int ret = o2.getSecond().compareTo(o1.getSecond());
        if (ret != 0) {
          return ret;
        }
        return o1.getFirst().compareTo(o2.getFirst());
      }
      
    });
    
    FPGrowth<Integer> fpGrowth = new FPGrowth<Integer>();
    fpGrowth.generateTopKFrequentPatterns(cTree.getIterator(), localFList, minSupport, maxHeapSize,
      new HashSet<Integer>(groupFeatures.get(key.get()).toList()), new IntegerStringOutputConverter(
          new ContextWriteOutputCollector<LongWritable,TransactionTree,Text,TopKStringPatterns>(context),
          featureReverseMap), new ContextStatusUpdater<LongWritable,TransactionTree,Text,TopKStringPatterns>(
          context));
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    
    super.setup(context);
    Parameters params = Parameters.fromString(context.getConfiguration().get("pfp.parameters", ""));
    
    int i = 0;
    for (Pair<String,Long> e : PFPGrowth.deserializeList(params, "fList", context.getConfiguration())) {
      featureReverseMap.add(e.getFirst());
      fMap.put(e.getFirst(), i);
      fRMap.add(e.getFirst());
      fList.add(new Pair<Integer,Long>(i++, e.getSecond()));
      
    }
    
    Map<String,Long> gList = PFPGrowth.deserializeMap(params, "gList", context.getConfiguration());
    
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
    maxHeapSize = Integer.valueOf(params.get("maxHeapSize", "50"));
    minSupport = Integer.valueOf(params.get("minSupport", "3"));
    FPTreeDepthCache.setFirstLevelCacheSize(Integer.valueOf(params.get("treeCacheSize", Integer
        .toString(FPTreeDepthCache.getFirstLevelCacheSize()))));
  }
}
