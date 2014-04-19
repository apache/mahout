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

package org.apache.mahout.fpm.pfpgrowth.fpgrowth;

import com.google.common.collect.Lists;

import java.util.List;

/**
 * Caches large FPTree {@link Object} for each level of the recursive
 * {@link FPGrowth} algorithm to reduce allocation overhead.
 */
public class FPTreeDepthCache {

  private final LeastKCache<Integer,FPTree> firstLevelCache = new LeastKCache<Integer,FPTree>(5);
  private int hits;
  private int misses;
  private final List<FPTree> treeCache = Lists.newArrayList();
  
  public final FPTree getFirstLevelTree(Integer attr) {
    FPTree tree = firstLevelCache.get(attr);
    if (tree != null) {
      hits++;
      return tree;
    } else {
      misses++;
      FPTree conditionalTree = new FPTree();
      firstLevelCache.set(attr, conditionalTree);
      return conditionalTree;
    }
  }
  
  public final int getHits() {
    return hits;
  }
  
  public final int getMisses() {
    return misses;
  }
  
  public final FPTree getTree(int level) {
    while (treeCache.size() < level + 1) {
      FPTree cTree = new FPTree();
      treeCache.add(cTree);
    }
    FPTree conditionalTree = treeCache.get(level);
    conditionalTree.clear();
    return conditionalTree;
  }
  
}
