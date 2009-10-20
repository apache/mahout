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

import java.util.ArrayList;

import org.apache.mahout.common.cache.LeastKCache;

public class FPTreeDepthCache {

  public static int firstLevelCacheSize = 5;

  private final ArrayList<FPTree> treeCache = new ArrayList<FPTree>();

  private final LeastKCache<Integer, FPTree> firstLevelCache = new LeastKCache<Integer, FPTree>(
      firstLevelCacheSize);

  public final FPTree getTree(int level) {
    while (treeCache.size() < level + 1) {
      FPTree cTree = new FPTree();
      treeCache.add(cTree);
    }
    FPTree conditionalTree = treeCache.get(level);
    conditionalTree.clear();
    return conditionalTree;
  }

  private int hits = 0;

  private int misses = 0;

  public final FPTree getFirstLevelTree(int attr) {
    Integer attribute = attr;
    if (firstLevelCache.contains(attribute)) {
      hits++;
      return firstLevelCache.get(attribute);
    } else {
      misses++;
      FPTree conditionalTree = new FPTree();
      firstLevelCache.set(attribute, conditionalTree);
      return conditionalTree;
    }
  }

  public final int getHits() {
    return hits;
  }

  public final int getMisses() {
    return misses;
  }

}
