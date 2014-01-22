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

package org.apache.mahout.fpm.pfpgrowth.fpgrowth2;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Maps;

import org.apache.commons.lang3.mutable.MutableLong;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.convertors.TopKPatternsOutputConverter;
import org.apache.mahout.math.list.LongArrayList;
import org.apache.mahout.math.list.IntArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import  org.apache.mahout.fpm.pfpgrowth.fpgrowth.Pattern;
import  org.apache.mahout.fpm.pfpgrowth.fpgrowth.FrequentPatternMaxHeap;
/**
 * Implementation of PFGrowth Algorithm
 */
public final class FPGrowthIds {

  private static final Logger log = LoggerFactory.getLogger(FPGrowthIds.class);

  private FPGrowthIds() {
  }

 /**
   * Generate Top K Frequent Patterns for every feature in returnableFeatures
   * given a stream of transactions and the minimum support
   *
   *
  * @param transactionStream
  *          Iterator of transaction
  * @param attributeFrequency
  *          list of frequent features and their support value
  * @param minSupport
  *          minimum support of the transactions
  * @param k
  *          Number of top frequent patterns to keep
  * @param returnableFeatures
  *          set of features for which the frequent patterns are mined. If the
  *          set is empty or null, then top K patterns for every frequent item (an item
  *          whose support> minSupport) is generated
  * @param output
  *          The output collector to which the the generated patterns are
  *          written
  * @throws IOException
   */
  public static void generateTopKFrequentPatterns(Iterator<Pair<IntArrayList, Long>> transactionStream,
                                                  LongArrayList attributeFrequency,
                                                  long minSupport,
                                                  int k,
                                                  IntArrayList returnableFeatures,
                                                  OutputCollector<Integer, List<Pair<List<Integer>, Long>>> output) throws IOException {

    for (int i = 0; i < attributeFrequency.size(); i++) {
      if (attributeFrequency.get(i) < minSupport) {
        attributeFrequency.setSize(i);
        attributeFrequency.trimToSize();
        break;
      }
    }

    log.info("Number of unique items {}", attributeFrequency.size());

    if (returnableFeatures == null || returnableFeatures.isEmpty()) {
      returnableFeatures = new IntArrayList();
      for (int j = 0; j < attributeFrequency.size(); j++) {
        returnableFeatures.add(j);
      }
    }

    log.info("Number of unique pruned items {}", attributeFrequency.size());
    generateTopKFrequentPatterns(transactionStream, attributeFrequency,
        minSupport, k, returnableFeatures,
        new TopKPatternsOutputConverter<Integer>(output, new IdentityMapping()));
  }

  private static class IdentityMapping extends AbstractMap<Integer, Integer> {

    @Override
    public Set<Map.Entry<Integer,Integer>> entrySet() {
      throw new IllegalStateException();
    }

    @Override
    public Integer get(Object key) {
      return (Integer) key;
    }

  }

  /**
   * Top K FpGrowth Algorithm
   *
   *
   * @param tree
   *          to be mined
   * @param minSupportValue
   *          minimum support of the pattern to keep
   * @param k
   *          Number of top frequent patterns to keep
   * @param requiredFeatures
   *          Set of integer id's of features to mine
   * @param outputCollector
   *          the Collector class which converts the given frequent pattern in
   *          integer to A
   * @return Top K Frequent Patterns for each feature and their support
   */
  private static Map<Integer,FrequentPatternMaxHeap> fpGrowth(FPTree tree,
                                                              long minSupportValue,
                                                              int k,
                                                              IntArrayList requiredFeatures,
                                                              TopKPatternsOutputConverter<Integer> outputCollector) throws IOException {

    Map<Integer,FrequentPatternMaxHeap> patterns = Maps.newHashMap();
    requiredFeatures.sort();
    for (int attribute : tree.attrIterableRev()) {
      if (requiredFeatures.binarySearch(attribute) >= 0) {
        log.info("Mining FTree Tree for all patterns with {}", attribute);
        MutableLong minSupport = new MutableLong(minSupportValue);
        FrequentPatternMaxHeap frequentPatterns = growth(tree, minSupport, k,
                                                         attribute);
        patterns.put(attribute, frequentPatterns);
        outputCollector.collect(attribute, frequentPatterns);

        minSupportValue = Math.max(minSupportValue, minSupport.longValue() / 2);
        log.info("Found {} Patterns with Least Support {}", patterns.get(
            attribute).count(), patterns.get(attribute).leastSupport());
      }
    }
    return patterns;
  }

      

  /**
   * Internal TopKFrequentPattern Generation algorithm, which represents the A's
   * as integers and transforms features to use only integers
   *
   * @param transactions
   *          Transaction database Iterator
   * @param attributeFrequency
   *          array representing the Frequency of the corresponding attribute id
   * @param minSupport
 *          minimum support of the pattern to be mined
   * @param k
*          Max value of the Size of the Max-Heap in which Patterns are held
   * @param returnFeatures
*          the id's of the features for which Top K patterns have to be mined
   * @param topKPatternsOutputCollector
*          the outputCollector which transforms the given Pattern in integer
   */
  private static void generateTopKFrequentPatterns(
      Iterator<Pair<IntArrayList, Long>> transactions,
      LongArrayList attributeFrequency,
      long minSupport,
      int k,
      IntArrayList returnFeatures,
      TopKPatternsOutputConverter<Integer> topKPatternsOutputCollector) throws IOException {

    FPTree tree = new FPTree(attributeFrequency, minSupport);

    // Constructing initial FPTree from the list of transactions
    int i = 0;
    while (transactions.hasNext()) {
      Pair<IntArrayList,Long> transaction = transactions.next();
      IntArrayList iArr = transaction.getFirst();
      tree.accumulate(iArr, transaction.getSecond());
      i++;
      if (i % 10000 == 0) {
        log.info("FPTree Building: Read {} Transactions", i);
      }
    }

    fpGrowth(tree, minSupport, k, returnFeatures, topKPatternsOutputCollector);
  }

  /** 
   * Run FP Growth recursively on tree, for the given target attribute
   */
  private static FrequentPatternMaxHeap growth(FPTree tree,
                                               MutableLong minSupportMutable,
                                               int k,
                                               int currentAttribute) {

    long currentAttributeCount = tree.headerCount(currentAttribute);

    if (currentAttributeCount < minSupportMutable.longValue()) {
      return new FrequentPatternMaxHeap(k, true);
    }
 
    FPTree condTree = tree.createMoreFreqConditionalTree(currentAttribute);

    Pair<FPTree, FPTree> pAndQ = condTree.splitSinglePrefix();
    FPTree p = pAndQ.getFirst();
    FPTree q = pAndQ.getSecond();

    FrequentPatternMaxHeap prefixPats = null;
    if (p != null) {
      prefixPats = mineSinglePrefix(p, k);
    }

    FrequentPatternMaxHeap suffixPats = new FrequentPatternMaxHeap(k, true);

    Pattern thisPat = new Pattern();
    thisPat.add(currentAttribute, currentAttributeCount);
    suffixPats.insert(thisPat);

    for (int attr : q.attrIterableRev())  {
      mergeHeap(suffixPats,
                growth(q, minSupportMutable, k, attr),
                currentAttribute,
                currentAttributeCount, true);
    }

    if (prefixPats != null) {
      return cross(prefixPats, suffixPats, k);
    }

    return suffixPats;
  }


  /** 
   * Return a set patterns which are the cross product of the patterns
   * in pPats and qPats.  
   */
  private static FrequentPatternMaxHeap cross(FrequentPatternMaxHeap pPats, 
                                              FrequentPatternMaxHeap qPats,
                                              int k) {
    FrequentPatternMaxHeap pats = new FrequentPatternMaxHeap(k, true);

    for (Pattern p : pPats.getHeap()) {
      int[] pints = p.getPattern();
      for (Pattern q : qPats.getHeap()) {
        int[] qints = q.getPattern();
        
        Pattern pq = new Pattern();
        for (int pi = 0; pi < p.length(); pi++) {
          pq.add(pints[pi], p.support());
        }
        for (int qi = 0; qi < q.length(); qi++) {
          pq.add(qints[qi], q.support());
        }
        pats.insert(pq);
      }
    }

    for (Pattern q : qPats.getHeap()) {
      Pattern qq = new Pattern();
      int[] qints = q.getPattern();
      for (int qi = 0; qi < q.length(); qi++) {
        qq.add(qints[qi], q.support());
      }
      pats.insert(qq);
    }

    return pats;
  }

  /**
   * Mine all frequent patterns that can be created by following a prefix
   * that is common to all sets in the given tree.
   */
  private static FrequentPatternMaxHeap mineSinglePrefix(FPTree tree, int k) {
    FrequentPatternMaxHeap pats = new FrequentPatternMaxHeap(k, true);
    FPTree.FPNode currNode = tree.root();
    while (currNode.numChildren() == 1) {
      currNode = currNode.children().iterator().next();
      FrequentPatternMaxHeap singlePat = new FrequentPatternMaxHeap(k, true);
      Pattern p = new Pattern();
      p.add(currNode.attribute(), currNode.count());
      singlePat.insert(p);
      pats = cross(singlePat, pats, k);
      pats.insert(p);
    }

    return pats;
  }

  private static void mergeHeap(FrequentPatternMaxHeap frequentPatterns,
                                FrequentPatternMaxHeap returnedPatterns,
                                int attribute,
                                long count,
                                boolean addAttribute) {
    frequentPatterns.addAll(returnedPatterns, attribute, count);
    if (frequentPatterns.addable(count) && addAttribute) {
      Pattern p = new Pattern();
      p.add(attribute, count);
      frequentPatterns.insert(p);
    }
  }
}

