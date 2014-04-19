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

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import org.apache.commons.lang3.mutable.MutableLong;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.fpm.pfpgrowth.CountDescendingPairComparator;
import org.apache.mahout.fpm.pfpgrowth.convertors.StatusUpdater;
import org.apache.mahout.fpm.pfpgrowth.convertors.TopKPatternsOutputConverter;
import org.apache.mahout.fpm.pfpgrowth.convertors.TransactionIterator;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.math.map.OpenIntIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of PFGrowth Algorithm with FP-Bonsai pruning
 *
 * @param <A> object type used as the cell items in a transaction list
 */
public class FPGrowth<A extends Comparable<? super A>> {

  private static final Logger log = LoggerFactory.getLogger(FPGrowth.class);

  public static List<Pair<String,TopKStringPatterns>> readFrequentPattern(Configuration conf, Path path) {
    List<Pair<String,TopKStringPatterns>> ret = Lists.newArrayList();
    // key is feature value is count
    for (Pair<Writable,TopKStringPatterns> record
         : new SequenceFileIterable<Writable,TopKStringPatterns>(path, true, conf)) {
      ret.add(new Pair<String,TopKStringPatterns>(record.getFirst().toString(),
                                                  new TopKStringPatterns(record.getSecond().getPatterns())));
    }
    return ret;
  }

  /**
   * Generate the Feature Frequency list from the given transaction whose
   * frequency > minSupport
   *
   * @param transactions
   *          Iterator over the transaction database
   * @param minSupport
   *          minSupport of the feature to be included
   * @return the List of features and their associated frequency as a Pair
   */
  public final List<Pair<A,Long>> generateFList(Iterator<Pair<List<A>,Long>> transactions, int minSupport) {

    Map<A,MutableLong> attributeSupport = Maps.newHashMap();
    while (transactions.hasNext()) {
      Pair<List<A>,Long> transaction = transactions.next();
      for (A attribute : transaction.getFirst()) {
        if (attributeSupport.containsKey(attribute)) {
          attributeSupport.get(attribute).add(transaction.getSecond().longValue());
        } else {
          attributeSupport.put(attribute, new MutableLong(transaction.getSecond()));
        }
      }
    }
    List<Pair<A,Long>> fList = Lists.newArrayList();
    for (Entry<A,MutableLong> e : attributeSupport.entrySet()) {
      long value = e.getValue().longValue();
      if (value >= minSupport) {
        fList.add(new Pair<A,Long>(e.getKey(), value));
      }
    }

    Collections.sort(fList, new CountDescendingPairComparator<A,Long>());

    return fList;
  }

  /**
   * Generate Top K Frequent Patterns for every feature in returnableFeatures
   * given a stream of transactions and the minimum support
   *
   * @param transactionStream
   *          Iterator of transaction
   * @param frequencyList
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
  public final void generateTopKFrequentPatterns(Iterator<Pair<List<A>,Long>> transactionStream,
                                                 Collection<Pair<A, Long>> frequencyList,
                                                 long minSupport,
                                                 int k,
                                                 Collection<A> returnableFeatures,
                                                 OutputCollector<A,List<Pair<List<A>,Long>>> output,
                                                 StatusUpdater updater) throws IOException {

    Map<Integer,A> reverseMapping = Maps.newHashMap();
    Map<A,Integer> attributeIdMapping = Maps.newHashMap();

    int id = 0;
    for (Pair<A,Long> feature : frequencyList) {
      A attrib = feature.getFirst();
      Long frequency = feature.getSecond();
      if (frequency >= minSupport) {
        attributeIdMapping.put(attrib, id);
        reverseMapping.put(id++, attrib);
      }
    }

    long[] attributeFrequency = new long[attributeIdMapping.size()];
    for (Pair<A,Long> feature : frequencyList) {
      A attrib = feature.getFirst();
      Long frequency = feature.getSecond();
      if (frequency < minSupport) {
        break;
      }
      attributeFrequency[attributeIdMapping.get(attrib)] = frequency;
    }

    log.info("Number of unique items {}", frequencyList.size());

    Collection<Integer> returnFeatures = Sets.newHashSet();
    if (returnableFeatures != null && !returnableFeatures.isEmpty()) {
      for (A attrib : returnableFeatures) {
        if (attributeIdMapping.containsKey(attrib)) {
          returnFeatures.add(attributeIdMapping.get(attrib));
          log.info("Adding Pattern {}=>{}", attrib, attributeIdMapping
            .get(attrib));
        }
      }
    } else {
      for (int j = 0; j < attributeIdMapping.size(); j++) {
        returnFeatures.add(j);
      }
    }

    log.info("Number of unique pruned items {}", attributeIdMapping.size());
    generateTopKFrequentPatterns(new TransactionIterator<A>(transactionStream,
        attributeIdMapping), attributeFrequency, minSupport, k, reverseMapping
        .size(), returnFeatures, new TopKPatternsOutputConverter<A>(output,
            reverseMapping), updater);

  }

  /**
   * Top K FpGrowth Algorithm
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
  private Map<Integer,FrequentPatternMaxHeap> fpGrowth(FPTree tree,
                                                       long minSupportValue,
                                                       int k,
                                                       Collection<Integer> requiredFeatures,
                                                       TopKPatternsOutputConverter<A> outputCollector,
                                                       StatusUpdater updater) throws IOException {

    Map<Integer,FrequentPatternMaxHeap> patterns = Maps.newHashMap();
    FPTreeDepthCache treeCache = new FPTreeDepthCache();
    for (int i = tree.getHeaderTableCount() - 1; i >= 0; i--) {
      int attribute = tree.getAttributeAtIndex(i);
      if (requiredFeatures.contains(attribute)) {
        log.info("Mining FTree Tree for all patterns with {}", attribute);
        MutableLong minSupport = new MutableLong(minSupportValue);
        FrequentPatternMaxHeap frequentPatterns = growth(tree, minSupport, k,
                                                         treeCache, 0, attribute, updater);
        patterns.put(attribute, frequentPatterns);
        outputCollector.collect(attribute, frequentPatterns);

        minSupportValue = Math.max(minSupportValue, minSupport.longValue() / 2);
        log.info("Found {} Patterns with Least Support {}", patterns.get(
            attribute).count(), patterns.get(attribute).leastSupport());
      }
    }
    log.info("Tree Cache: First Level: Cache hits={} Cache Misses={}",
      treeCache.getHits(), treeCache.getMisses());
    return patterns;
  }

  private static FrequentPatternMaxHeap generateSinglePathPatterns(FPTree tree,
                                                                   int k,
                                                                   long minSupport) {
    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(k, false);

    int tempNode = FPTree.ROOTNODEID;
    Pattern frequentItem = new Pattern();
    while (tree.childCount(tempNode) != 0) {
      if (tree.childCount(tempNode) > 1) {
        log.info("This should not happen {} {}", tree.childCount(tempNode),
          tempNode);
      }
      tempNode = tree.childAtIndex(tempNode, 0);
      if (tree.count(tempNode) >= minSupport) {
        frequentItem.add(tree.attribute(tempNode), tree.count(tempNode));
      }
    }
    if (frequentItem.length() > 0) {
      frequentPatterns.insert(frequentItem);
    }

    return frequentPatterns;
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
   * @param featureSetSize
   *          number of features
   * @param returnFeatures
   *          the id's of the features for which Top K patterns have to be mined
   * @param topKPatternsOutputCollector
   *          the outputCollector which transforms the given Pattern in integer
   *          format to the corresponding A Format
   */
  private void generateTopKFrequentPatterns(
    Iterator<Pair<int[],Long>> transactions,
    long[] attributeFrequency,
    long minSupport,
    int k,
    int featureSetSize,
    Collection<Integer> returnFeatures, TopKPatternsOutputConverter<A> topKPatternsOutputCollector,
    StatusUpdater updater) throws IOException {

    FPTree tree = new FPTree(featureSetSize);
    for (int i = 0; i < featureSetSize; i++) {
      tree.addHeaderCount(i, attributeFrequency[i]);
    }

    // Constructing initial FPTree from the list of transactions
    int nodecount = 0;
    // int attribcount = 0;
    int i = 0;
    while (transactions.hasNext()) {
      Pair<int[],Long> transaction = transactions.next();
      Arrays.sort(transaction.getFirst());
      // attribcount += transaction.length;
      nodecount += treeAddCount(tree, transaction.getFirst(), transaction.getSecond(), minSupport, attributeFrequency);
      i++;
      if (i % 10000 == 0) {
        log.info("FPTree Building: Read {} Transactions", i);
      }
    }

    log.info("Number of Nodes in the FP Tree: {}", nodecount);

    fpGrowth(tree, minSupport, k, returnFeatures, topKPatternsOutputCollector, updater);
  }

  private static FrequentPatternMaxHeap growth(FPTree tree,
                                               MutableLong minSupportMutable,
                                               int k,
                                               FPTreeDepthCache treeCache,
                                               int level,
                                               int currentAttribute,
                                               StatusUpdater updater) {

    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(k,
      true);

    int i = Arrays.binarySearch(tree.getHeaderTableAttributes(),
      currentAttribute);
    if (i < 0) {
      return frequentPatterns;
    }

    int headerTableCount = tree.getHeaderTableCount();

    while (i < headerTableCount) {
      int attribute = tree.getAttributeAtIndex(i);
      long count = tree.getHeaderSupportCount(attribute);
      if (count < minSupportMutable.longValue()) {
        i++;
        continue;
      }
      updater.update("FPGrowth Algorithm for a given feature: " + attribute);
      FPTree conditionalTree = treeCache.getFirstLevelTree(attribute);
      if (conditionalTree.isEmpty()) {
        traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
          minSupportMutable.longValue(), conditionalTree, tree);
        // printTree(conditionalTree);

      }

      FrequentPatternMaxHeap returnedPatterns;
      if (attribute == currentAttribute) {

        returnedPatterns = growthTopDown(conditionalTree, minSupportMutable, k,
          treeCache, level + 1, true, currentAttribute, updater);

        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
          attribute, count, true);
      } else {
        returnedPatterns = growthTopDown(conditionalTree, minSupportMutable, k,
          treeCache, level + 1, false, currentAttribute, updater);
        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
          attribute, count, false);
      }
      if (frequentPatterns.isFull() && minSupportMutable.longValue() < frequentPatterns.leastSupport()) {
        minSupportMutable.setValue(frequentPatterns.leastSupport());
      }
      i++;
    }

    return frequentPatterns;
  }

  private static FrequentPatternMaxHeap growthBottomUp(FPTree tree,
                                                       MutableLong minSupportMutable,
                                                       int k,
                                                       FPTreeDepthCache treeCache,
                                                       int level,
                                                       boolean conditionalOfCurrentAttribute,
                                                       int currentAttribute,
                                                       StatusUpdater updater) {

    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(k,
      false);

    if (!conditionalOfCurrentAttribute) {
      int index = Arrays.binarySearch(tree.getHeaderTableAttributes(),
        currentAttribute);
      if (index < 0) {
        return frequentPatterns;
      } else {
        int attribute = tree.getAttributeAtIndex(index);
        long count = tree.getHeaderSupportCount(attribute);
        if (count < minSupportMutable.longValue()) {
          return frequentPatterns;
        }
      }
    }

    if (tree.singlePath()) {
      return generateSinglePathPatterns(tree, k, minSupportMutable.longValue());
    }

    updater.update("Bottom Up FP Growth");
    for (int i = tree.getHeaderTableCount() - 1; i >= 0; i--) {
      int attribute = tree.getAttributeAtIndex(i);
      long count = tree.getHeaderSupportCount(attribute);
      if (count < minSupportMutable.longValue()) {
        continue;
      }
      FPTree conditionalTree = treeCache.getTree(level);

      FrequentPatternMaxHeap returnedPatterns;
      if (conditionalOfCurrentAttribute) {
        traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
          minSupportMutable.longValue(), conditionalTree, tree);
        returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
          k, treeCache, level + 1, true, currentAttribute, updater);

        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
          attribute, count, true);
      } else {
        if (attribute == currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
            minSupportMutable.longValue(), conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
            k, treeCache, level + 1, true, currentAttribute, updater);

          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, true);
        } else if (attribute > currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
            minSupportMutable.longValue(), conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
            k, treeCache, level + 1, false, currentAttribute, updater);
          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, false);
        }
      }

      if (frequentPatterns.isFull() && minSupportMutable.longValue() < frequentPatterns.leastSupport()) {
        minSupportMutable.setValue(frequentPatterns.leastSupport());
      }
    }

    return frequentPatterns;
  }

  private static FrequentPatternMaxHeap growthTopDown(FPTree tree,
                                                      MutableLong minSupportMutable,
                                                      int k,
                                                      FPTreeDepthCache treeCache,
                                                      int level,
                                                      boolean conditionalOfCurrentAttribute,
                                                      int currentAttribute,
                                                      StatusUpdater updater) {

    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(k,
      true);

    if (!conditionalOfCurrentAttribute) {
      int index = Arrays.binarySearch(tree.getHeaderTableAttributes(),
        currentAttribute);
      if (index < 0) {
        return frequentPatterns;
      } else {
        int attribute = tree.getAttributeAtIndex(index);
        long count = tree.getHeaderSupportCount(attribute);
        if (count < minSupportMutable.longValue()) {
          return frequentPatterns;
        }
      }
    }

    if (tree.singlePath()) {
      return generateSinglePathPatterns(tree, k, minSupportMutable.longValue());
    }

    updater.update("Top Down Growth:");

    for (int i = 0; i < tree.getHeaderTableCount(); i++) {
      int attribute = tree.getAttributeAtIndex(i);
      long count = tree.getHeaderSupportCount(attribute);
      if (count < minSupportMutable.longValue()) {
        continue;
      }

      FPTree conditionalTree = treeCache.getTree(level);

      FrequentPatternMaxHeap returnedPatterns;
      if (conditionalOfCurrentAttribute) {
        traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
          minSupportMutable.longValue(), conditionalTree, tree);

        returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
          k, treeCache, level + 1, true, currentAttribute, updater);
        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
          attribute, count, true);

      } else {
        if (attribute == currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
            minSupportMutable.longValue(), conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
            k, treeCache, level + 1, true, currentAttribute, updater);
          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, true);

        } else if (attribute > currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
            minSupportMutable.longValue(), conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
            k, treeCache, level + 1, false, currentAttribute, updater);
          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, false);

        }
      }
      if (frequentPatterns.isFull() && minSupportMutable.longValue() < frequentPatterns.leastSupport()) {
        minSupportMutable.setValue(frequentPatterns.leastSupport());
      }
    }

    return frequentPatterns;
  }

  private static FrequentPatternMaxHeap mergeHeap(FrequentPatternMaxHeap frequentPatterns,
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

    return frequentPatterns;
  }

  private static void traverseAndBuildConditionalFPTreeData(int firstConditionalNode,
                                                            long minSupport,
                                                            FPTree conditionalTree,
                                                            FPTree tree) {

    // Build Subtable
    int conditionalNode = firstConditionalNode;

    while (conditionalNode != -1) {
      long nextNodeCount = tree.count(conditionalNode);
      int pathNode = tree.parent(conditionalNode);
      int prevConditional = -1;

      while (pathNode != 0) { // dummy root node
        int attribute = tree.attribute(pathNode);
        if (tree.getHeaderSupportCount(attribute) < minSupport) {
          pathNode = tree.parent(pathNode);
          continue;
        }
        // update and increment the headerTable Counts
        conditionalTree.addHeaderCount(attribute, nextNodeCount);

        int conditional = tree.conditional(pathNode);
        // if its a new conditional tree node

        if (conditional == 0) {
          tree.setConditional(pathNode, conditionalTree.createConditionalNode(
            attribute, 0));
          conditional = tree.conditional(pathNode);
          conditionalTree.addHeaderNext(attribute, conditional);
        } else {
          conditionalTree.setSinglePath(false);
        }

        if (prevConditional != -1) { // if there is a child element
          int prevParent = conditionalTree.parent(prevConditional);
          if (prevParent == -1) {
            conditionalTree.setParent(prevConditional, conditional);
          } else if (prevParent != conditional) {
            throw new IllegalStateException();
          }
        }

        conditionalTree.addCount(conditional, nextNodeCount);
        prevConditional = conditional;

        pathNode = tree.parent(pathNode);

      }

      if (prevConditional != -1) {
        int prevParent = conditionalTree.parent(prevConditional);
        if (prevParent == -1) {
          conditionalTree.setParent(prevConditional, FPTree.ROOTNODEID);
        } else if (prevParent != FPTree.ROOTNODEID) {
          throw new IllegalStateException();
        }
        if (conditionalTree.childCount(FPTree.ROOTNODEID) > 1 && conditionalTree.singlePath()) {
          conditionalTree.setSinglePath(false);
        }
      }
      conditionalNode = tree.next(conditionalNode);
    }

    tree.clearConditional();
    conditionalTree.reorderHeaderTable();
    pruneFPTree(minSupport, conditionalTree);
    // prune Conditional Tree

  }

  private static void pruneFPTree(long minSupport, FPTree tree) {
    for (int i = 0; i < tree.getHeaderTableCount(); i++) {
      int currentAttribute = tree.getAttributeAtIndex(i);
      if (tree.getHeaderSupportCount(currentAttribute) < minSupport) {
        int nextNode = tree.getHeaderNext(currentAttribute);
        tree.removeHeaderNext(currentAttribute);
        while (nextNode != -1) {

          int mychildCount = tree.childCount(nextNode);

          int parentNode = tree.parent(nextNode);

          for (int j = 0; j < mychildCount; j++) {
            Integer myChildId = tree.childAtIndex(nextNode, j);
            tree.replaceChild(parentNode, nextNode, myChildId);
          }
          nextNode = tree.next(nextNode);
        }

      }
    }

    for (int i = 0; i < tree.getHeaderTableCount(); i++) {
      int currentAttribute = tree.getAttributeAtIndex(i);
      int nextNode = tree.getHeaderNext(currentAttribute);

      OpenIntIntHashMap prevNode = new OpenIntIntHashMap();
      int justPrevNode = -1;
      while (nextNode != -1) {

        int parent = tree.parent(nextNode);

        if (prevNode.containsKey(parent)) {
          int prevNodeId = prevNode.get(parent);
          if (tree.childCount(prevNodeId) <= 1 && tree.childCount(nextNode) <= 1) {
            tree.addCount(prevNodeId, tree.count(nextNode));
            tree.addCount(nextNode, -1 * tree.count(nextNode));
            if (tree.childCount(nextNode) == 1) {
              tree.addChild(prevNodeId, tree.childAtIndex(nextNode, 0));
              tree.setParent(tree.childAtIndex(nextNode, 0), prevNodeId);
            }
            tree.setNext(justPrevNode, tree.next(nextNode));
          }
        } else {
          prevNode.put(parent, nextNode);
        }
        justPrevNode = nextNode;
        nextNode = tree.next(nextNode);
      }
    }

    // prune Conditional Tree

  }

  /**
   * Create FPTree with node counts incremented by addCount variable given the
   * root node and the List of Attributes in transaction sorted by support
   *
   * @param tree
   *          object to which the transaction has to be added to
   * @param myList
   *          List of transactions sorted by support
   * @param addCount
   *          amount by which the Node count has to be incremented
   * @param minSupport
   *          the MutableLong value which contains the current value(dynamic) of
   *          support
   * @param attributeFrequency
   *          the list of attributes and their frequency
   * @return the number of new nodes added
   */
  private static int treeAddCount(FPTree tree,
                                  int[] myList,
                                  long addCount,
                                  long minSupport,
                                  long[] attributeFrequency) {

    int temp = FPTree.ROOTNODEID;
    int ret = 0;
    boolean addCountMode = true;

    for (int attribute : myList) {
      if (attributeFrequency[attribute] < minSupport) {
        return ret;
      }
      int child;
      if (addCountMode) {
        child = tree.childWithAttribute(temp, attribute);
        if (child == -1) {
          addCountMode = false;
        } else {
          tree.addCount(child, addCount);
          temp = child;
        }
      }
      if (!addCountMode) {
        child = tree.createNode(temp, attribute, addCount);
        temp = child;
        ret++;
      }
    }

    return ret;

  }
}
