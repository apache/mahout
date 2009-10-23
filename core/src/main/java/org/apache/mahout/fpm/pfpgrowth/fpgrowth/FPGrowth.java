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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.Map.Entry;
import org.apache.commons.lang.mutable.MutableLong;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.convertors.TopKPatternsOutputConvertor;
import org.apache.mahout.fpm.pfpgrowth.convertors.TransactionIterator;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PFPGrowth Class has both vanilla FPGrowth and Top K FPGrowth
 * 
 * @param <A>
 */
public class FPGrowth<A extends Comparable<? super A>> {

  private static final Logger log = LoggerFactory.getLogger(FPGrowth.class);

  public static List<Pair<String, TopKStringPatterns>> readFrequentPattern(
      FileSystem fs, Configuration conf, Path path) throws IOException {

    List<Pair<String, TopKStringPatterns>> ret = new ArrayList<Pair<String, TopKStringPatterns>>();
    Text key = new Text();
    TopKStringPatterns value = new TopKStringPatterns();
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    // key is feature value is count
    while (reader.next(key, value)) {
      ret.add(new Pair<String, TopKStringPatterns>(key.toString(),
          new TopKStringPatterns(value.getPatterns())));
    }
    return ret;
  }

  /**
   * Generate the Feature Frequency list from the given transaction whose
   * frequency > minSupport
   * 
   * @param transactions Iterator over the transaction database
   * @param minSupport minSupport of the feature to be included
   * @return the List of features and their associated frequency as a Pair
   * @throws IOException
   */
  public final List<Pair<A, Long>> generateFList(
      Iterator<List<A>> transactions, int minSupport) {

    Map<A, MutableLong> AttributeSupport = new HashMap<A, MutableLong>();
    //int count = 0;
    while (transactions.hasNext()) {
      List<A> transaction = transactions.next();
      for (A attribute : transaction) {
        if (AttributeSupport.containsKey(attribute) == false)
          AttributeSupport.put(attribute, new MutableLong(1));
        else
          AttributeSupport.get(attribute).increment();
        //count++;
      }
    }
    List<Pair<A, Long>> fList = new ArrayList<Pair<A, Long>>();
    for (Entry<A, MutableLong> e : AttributeSupport.entrySet())
      fList.add(new Pair<A, Long>(e.getKey(), e.getValue()
          .longValue()));
    Collections.sort(fList, new Comparator<Pair<A, Long>>() {

      @Override
      public int compare(Pair<A, Long> o1,
          Pair<A, Long> o2) {
        int ret = o2.getSecond().compareTo(o1.getSecond());
        if (ret != 0)
          return ret;
        return o1.getFirst().compareTo(o2.getFirst());
      }

    });

    return fList;
  }

  /**
   * Generate Top K Frequent Patterns for every feature in returnableFeatures
   * given a stream of transactions and the minimum support
   * 
   * @param transactionStream Iterator of transaction
   * @param frequencyList list of frequent features and their support value
   * @param minSupport minimum support of the transactions
   * @param K Number of top frequent patterns to keep
   * @param returnableFeatures set of features for which the frequent patterns
   *        are mined. If the set is null, then top K patterns for every
   *        frequent item (an item whose support> minSupport) is generated
   * @param output The output collector to which the the generated patterns are
   *        written
   * @throws IOException
   */
  public final void generateTopKFrequentPatterns(
      Iterator<List<A>> transactionStream,
      List<Pair<A, Long>> frequencyList,
      long minSupport,
      int K,
      Set<A> returnableFeatures,
      OutputCollector<A, List<Pair<List<A>, Long>>> output)
      throws IOException {

    Map<Integer, A> reverseMapping = new HashMap<Integer, A>();
    Map<A, Integer> attributeIdMapping = new HashMap<A, Integer>();

    int id = 0;
    for (Pair<A, Long> feature : frequencyList) {
      A attrib = feature.getFirst();
      Long frequency = feature.getSecond();
      if (frequency < minSupport)
        continue;
      attributeIdMapping.put(attrib, id);
      reverseMapping.put(id++, attrib);
    }

    long[] attributeFrequency = new long[attributeIdMapping.size()];
    for (Pair<A, Long> feature : frequencyList) {
      A attrib = feature.getFirst();
      Long frequency = feature.getSecond();
      if (frequency < minSupport)
        break;
      attributeFrequency[attributeIdMapping.get(attrib)] = frequency;
    }

    log.info("Number of unique items {}", frequencyList.size());

    Set<Integer> returnFeatures = new HashSet<Integer>();
    if (returnableFeatures.isEmpty() == false) {
      for (A attrib : returnableFeatures) {
        if (attributeIdMapping.containsKey(attrib)) {
          returnFeatures.add(attributeIdMapping.get(attrib));
          log.info("Adding Pattern {}=>{}", attrib, attributeIdMapping
              .get(attrib));
        }
      }
    } else {
      for (int j = 0; j < attributeIdMapping.size(); j++)
        returnFeatures.add(j);
    }

    log.info("Number of unique pruned items {}", attributeIdMapping.size());
    generateTopKFrequentPatterns(new TransactionIterator<A>(
        transactionStream, attributeIdMapping), attributeFrequency, minSupport,
        K, reverseMapping.size(), returnFeatures,
        new TopKPatternsOutputConvertor<A>(output,
            reverseMapping));

  }

  /**
   * Top K FpGrowth Algorithm
   * 
   * @param tree to be mined
   * @param minSupportMutable minimum support of the pattern to keep
   * @param K Number of top frequent patterns to keep
   * @param requiredFeatures Set of integer id's of features to mine
   * @param outputCollector the Collector class which converts the given
   *        frequent pattern in integer to A
   * @return Top K Frequent Patterns for each feature and their support
   * @throws IOException
   */
  private Map<Integer, FrequentPatternMaxHeap> fpGrowth(FPTree tree,
      MutableLong minSupportMutable, int K, Set<Integer> requiredFeatures,
      TopKPatternsOutputConvertor<A> outputCollector)
      throws IOException {

    int minSupportValue = minSupportMutable.intValue();

    Map<Integer, FrequentPatternMaxHeap> Patterns = new HashMap<Integer, FrequentPatternMaxHeap>();
    FPTreeDepthCache treeCache = new FPTreeDepthCache();
    for (int i = tree.getHeaderTableCount() - 1; i >= 0; i--) {
      int attribute = tree.getAttributeAtIndex(i);
      if (requiredFeatures.contains(attribute) == false)
        continue;
      log.info("Mining FTree Tree for all patterns with {}", attribute);
      MutableLong minSupport = new MutableLong(minSupportValue);
      FrequentPatternMaxHeap frequentPatterns = growth(tree, minSupport, K,
          treeCache, 0, attribute);
      Patterns.put(attribute, frequentPatterns);
      outputCollector.collect(attribute, frequentPatterns);

      minSupportValue = Math.max(minSupportValue, minSupport.intValue() / 2);
      log.info("Found {} Patterns with Least Support {}", Patterns.get(
          attribute).count(), Patterns.get(attribute).leastSupport());
    }
    log.info("Tree Cache: First Level: Cache hits={} Cache Misses={}",
        treeCache.getHits(), treeCache.getMisses());
    return Patterns;
  }

  private static FrequentPatternMaxHeap generateSinglePathPatterns(FPTree tree, int K,
      MutableLong minSupportMutable) {
    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(K);

    int tempNode = FPTree.ROOTNODEID;
    Pattern frequentItem = new Pattern();
    while (tree.childCount(tempNode) != 0) {
      if (tree.childCount(tempNode) > 1) {
        log.info("This should not happen {} {}", tree.childCount(tempNode),
            tempNode);
      }
      tempNode = tree.childAtIndex(tempNode, 0);
      if (tree.count(tempNode) < minSupportMutable.intValue()) {
        continue;
      }
      frequentItem.add(tree.attribute(tempNode), tree.count(tempNode));
    }
    if (frequentItem.length() > 0) {
      frequentPatterns.insert(frequentItem);
    }

    return frequentPatterns;
  }

  /**
   * Internal TopKFrequentPattern Generation algorithm, which represents the
   * A's as integers and transforms features to use only
   * integers
   * 
   * @param transactions Transaction database Iterator
   * @param attributeFrequency array representing the Frequency of the
   *        corresponding attribute id
   * @param minSupport minimum support of the pattern to be mined
   * @param K Max value of the Size of the Max-Heap in which Patterns are held
   * @param featureSetSize number of features
   * @param returnFeatures the id's of the features for which Top K patterns
   *        have to be mined
   * @param topKPatternsOutputCollector the outputCollector which transforms the
   *        given Pattern in integer format to the corresponding
   *        A Format
   * @return Top K frequent patterns for each attribute
   * @throws IOException
   */
  private Map<Integer, FrequentPatternMaxHeap> generateTopKFrequentPatterns(
      Iterator<int[]> transactions,
      long[] attributeFrequency,
      long minSupport,
      int K,
      int featureSetSize,
      Set<Integer> returnFeatures,
      TopKPatternsOutputConvertor<A> topKPatternsOutputCollector)
      throws IOException {

    FPTree tree = new FPTree(featureSetSize);
    for (int i = 0; i < featureSetSize; i++) {
      tree.addHeaderCount(i, attributeFrequency[i]);
    }

    // Constructing initial FPTree from the list of transactions
    MutableLong minSupportMutable = new MutableLong(minSupport);
    int nodecount = 0;
    //int attribcount = 0;
    int i = 0;
    while (transactions.hasNext()) {
      int[] transaction = transactions.next();
      Arrays.sort(transaction);
      //attribcount += transaction.length;
      nodecount += treeAddCount(tree, transaction, 1, minSupportMutable,
          attributeFrequency);
      i++;
      if (i % 10000 == 0) {
        log.info("FPTree Building: Read {} Transactions", i);
      }
    }

    log.info("Number of Nodes in the FP Tree: {}", nodecount);

    return fpGrowth(tree, minSupportMutable, K, returnFeatures,
        topKPatternsOutputCollector);
  }

  private FrequentPatternMaxHeap growth(FPTree tree,
      MutableLong minSupportMutable, int K, FPTreeDepthCache treeCache,
      int level, int currentAttribute) {
    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(K);

    int i = Arrays.binarySearch(tree.getHeaderTableAttributes(),
        currentAttribute);
    if (i < 0)
      return frequentPatterns;

    FrequentPatternMaxHeap returnedPatterns;
    int headerTableCount = tree.getHeaderTableCount();
    while (i < headerTableCount) {
      int attribute = tree.getAttributeAtIndex(i);
      long count = tree.getHeaderSupportCount(attribute);
      if (count < minSupportMutable.intValue()){
	i++;
	continue;
      }

      FPTree conditionalTree = treeCache.getFirstLevelTree(attribute);
      if (conditionalTree.isEmpty()) {
        traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
            minSupportMutable, conditionalTree, tree);
      }

      if (attribute == currentAttribute) {

        returnedPatterns = growthTopDown(conditionalTree, minSupportMutable, K,
            treeCache, level + 1, true, currentAttribute);
        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, true, true);
      } else {
        returnedPatterns = growthTopDown(conditionalTree, minSupportMutable, K,
            treeCache, level + 1, false, currentAttribute);

        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, false, true);
      }
      if (frequentPatterns.isFull()) {
        if (minSupportMutable.intValue() < frequentPatterns.leastSupport()) {
          minSupportMutable.setValue(frequentPatterns.leastSupport());
        }
      }
      i++;
    }

    return frequentPatterns;
  }

  private static FrequentPatternMaxHeap growthBottomUp(FPTree tree,
      MutableLong minSupportMutable, int K, FPTreeDepthCache treeCache,
      int level, boolean conditionalOfCurrentAttribute, int currentAttribute) {
    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(K);
    if (conditionalOfCurrentAttribute == false) {
      int index = Arrays.binarySearch(tree.getHeaderTableAttributes(),
          currentAttribute);
      if (index < 0)
        return frequentPatterns;
      else {
        int attribute = tree.getAttributeAtIndex(index);
        long count = tree.getHeaderSupportCount(attribute);
        if (count < minSupportMutable.longValue())
          return frequentPatterns;
      }
    }
    if (tree.singlePath()) {
      return generateSinglePathPatterns(tree, K, minSupportMutable);
    }
    FrequentPatternMaxHeap returnedPatterns;
    for (int i = tree.getHeaderTableCount() - 1; i >= 0; i--) {
      int attribute = tree.getAttributeAtIndex(i);
      long count = tree.getHeaderSupportCount(attribute);
      if (count < minSupportMutable.longValue())
        continue;
      FPTree conditionalTree = treeCache.getTree(level);

      if (conditionalOfCurrentAttribute) {
        traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
            minSupportMutable, conditionalTree, tree);
        returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
            K, treeCache, level + 1, true, currentAttribute);

        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, true, false);
      } else {
        if (attribute == currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
              minSupportMutable, conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
              K, treeCache, level + 1, true, currentAttribute);

          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
              attribute, count, true, false);
        } else if (attribute > currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
              minSupportMutable, conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
              K, treeCache, level + 1, false, currentAttribute);
          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
              attribute, count, false, false);
        }
      }

      if (frequentPatterns.isFull()) {
        if (minSupportMutable.intValue() < frequentPatterns.leastSupport()) {
          minSupportMutable.setValue(frequentPatterns.leastSupport());
        }
      }
    }

    return frequentPatterns;
  }

  private FrequentPatternMaxHeap growthTopDown(FPTree tree,
      MutableLong minSupportMutable, int K, FPTreeDepthCache treeCache,
      int level, boolean conditionalOfCurrentAttribute, int currentAttribute) {
    FrequentPatternMaxHeap frequentPatterns = new FrequentPatternMaxHeap(K);
    if (conditionalOfCurrentAttribute == false) {
      int index = Arrays.binarySearch(tree.getHeaderTableAttributes(),
          currentAttribute);
      if (index < 0)
        return frequentPatterns;
      else {
        int attribute = tree.getAttributeAtIndex(index);
        long count = tree.getHeaderSupportCount(attribute);
        if (count < minSupportMutable.intValue())
          return frequentPatterns;
      }
    }
    if (tree.singlePath()) {
      return generateSinglePathPatterns(tree, K, minSupportMutable);
    }
    FrequentPatternMaxHeap returnedPatterns;
    for (int i = 0; i < tree.getHeaderTableCount(); i++) {
      int attribute = tree.getAttributeAtIndex(i);
      long count = tree.getHeaderSupportCount(attribute);
      if (count < minSupportMutable.longValue())
        continue;

      FPTree conditionalTree = treeCache.getTree(level);

      if (conditionalOfCurrentAttribute) {
        traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
            minSupportMutable, conditionalTree, tree);
        returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
            K, treeCache, level + 1, true, currentAttribute);
        frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
            attribute, count, true, true);

      } else {
        if (attribute == currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
              minSupportMutable, conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
              K, treeCache, level + 1, true, currentAttribute);
          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
              attribute, count, true, false);

        } else if (attribute > currentAttribute) {
          traverseAndBuildConditionalFPTreeData(tree.getHeaderNext(attribute),
              minSupportMutable, conditionalTree, tree);
          returnedPatterns = growthBottomUp(conditionalTree, minSupportMutable,
              K, treeCache, level + 1, false, currentAttribute);
          frequentPatterns = mergeHeap(frequentPatterns, returnedPatterns,
              attribute, count, false, true);

        }
      }
      if (frequentPatterns.isFull()) {
        if (minSupportMutable.intValue() < frequentPatterns.leastSupport()) {
          minSupportMutable.setValue(frequentPatterns.leastSupport());
        }
      }
    }

    return frequentPatterns;
  }

  private static FrequentPatternMaxHeap mergeHeap(
      FrequentPatternMaxHeap frequentPatterns,
      FrequentPatternMaxHeap returnedPatterns, int attribute, long count,
      boolean addAttribute, boolean subPatternCheck) {

    Set<Pattern> myHeap = returnedPatterns.getHeap();

    for (Pattern pattern : myHeap) {
      long support = Math.min(count, pattern.support());
      if (frequentPatterns.addable(support)) {
        pattern.add(attribute, count);
        frequentPatterns.insert(pattern, subPatternCheck);
      }
    }
    if (frequentPatterns.addable(count) && addAttribute) {
      Pattern p = new Pattern();
      p.add(attribute, count);
      frequentPatterns.insert(p, subPatternCheck);
    }
    return frequentPatterns;
  }

  private static void traverseAndBuildConditionalFPTreeData(int firstConditionalNode,
      MutableLong minSupportMutable, FPTree conditionalTree, FPTree tree) {

    // Build Subtable
    int conditionalNode = firstConditionalNode;

    while (conditionalNode != -1) {
      long nextNodeCount = tree.count(conditionalNode);
      int pathNode = tree.parent(conditionalNode);
      int prevConditional = -1;

      while (pathNode != 0) { // dummy root node
        int attribute = tree.attribute(pathNode);
        if (tree.getHeaderSupportCount(attribute) < minSupportMutable
            .intValue()) {
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
          conditionalTree.setParent(prevConditional, conditional);
        }

        conditionalTree.addCount(conditional, nextNodeCount);
        prevConditional = conditional;

        pathNode = tree.parent(pathNode);

      }
      if (prevConditional != -1) {
        conditionalTree.setParent(prevConditional, FPTree.ROOTNODEID);
        if (conditionalTree.childCount(FPTree.ROOTNODEID) > 1
            && conditionalTree.singlePath()) {
          conditionalTree.setSinglePath(false);

        }
      }
      conditionalNode = tree.next(conditionalNode);
    }
    tree.clearConditional();
    conditionalTree.reorderHeaderTable();

  }

  /**
   * Create FPTree with node counts incremented by addCount variable given the
   * root node and the List of Attributes in transaction sorted by support
   * 
   * @param tree object to which the transaction has to be added to
   * @param myList List of transactions sorted by support
   * @param addCount amount by which the Node count has to be incremented
   * @param minSupport the MutableLong value which contains the current
   *        value(dynamic) of support
   * @param attributeFrequency the list of attributes and their frequency
   * @return the number of new nodes added
   */
  private static int treeAddCount(FPTree tree, int[] myList, int addCount,
      MutableLong minSupport, long[] attributeFrequency) {
    int temp = FPTree.ROOTNODEID;
    int ret = 0;
    boolean addCountMode = true;
    for (int attribute : myList) {
      if (attributeFrequency[attribute] < minSupport.intValue())
        return ret;
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
