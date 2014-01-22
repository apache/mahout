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

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import com.google.common.collect.Lists;

import org.apache.mahout.common.Pair;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.list.LongArrayList;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

/**
 * A straightforward implementation of FPTrees as described in Han et. al.
 */
public final class FPTree {

  private final AttrComparator attrComparator = new AttrComparator();
  private final FPNode root;
  private final long minSupport;
  private final LongArrayList attrCountList;
  private final OpenIntObjectHashMap<List<FPNode>> attrNodeLists;

  public static final class FPNode {
    private final FPNode parent;
    private final OpenIntObjectHashMap<FPNode> childMap;
    private final int attribute;
    private long count;

    private FPNode(FPNode parent, int attribute, long count) {
      this.parent = parent;
      this.attribute = attribute;
      this.count = count;
      this.childMap = new OpenIntObjectHashMap<FPNode>();
    }

    private void addChild(FPNode child) {
      this.childMap.put(child.attribute(), child);
    }

    public Iterable<FPNode> children() {
      return childMap.values();
    }

    public int numChildren() {
      return childMap.size();
    }

    public FPNode parent() {
      return parent;
    }

    public FPNode child(int attribute) {
      return childMap.get(attribute);
    }

    public int attribute() {
      return attribute;
    }

    public void accumulate(long incr) {
      count += incr;
    }

    public long count() {
      return count;
    }

  }

  /**
   * Creates an FPTree using the attribute counts in attrCountList.
   *
   * Note that the counts in attrCountList are assumed to be complete;
   * they are not updated as the tree is modified.
   */
  public FPTree(LongArrayList attrCountList, long minSupport) {
    this.root = new FPNode(null, -1, 0);
    this.attrCountList = attrCountList;
    this.attrNodeLists = new OpenIntObjectHashMap<List<FPNode>>();
    this.minSupport = minSupport;
  }

  /**
   * Creates an FPTree using the attribute counts in attrCounts.
   *
   * Note that the counts in attrCounts are assumed to be complete;
   * they are not updated as the tree is modified.
   */
  public FPTree(long[] attrCounts, long minSupport) {
    this.root = new FPNode(null, -1, 0);
    this.attrCountList = new LongArrayList();
    for (int i = 0; i < attrCounts.length; i++) {
      if (attrCounts[i] > 0) {
        if (attrCountList.size() < (i + 1)) {
          attrCountList.setSize(i + 1);
        }
        attrCountList.set(i, attrCounts[i]);
      }
    }
    this.attrNodeLists = new OpenIntObjectHashMap<List<FPNode>>();
    this.minSupport = minSupport;
  }


  /**
   * Returns the count of the given attribute, as supplied on construction.
   */
  public long headerCount(int attribute) {
    return attrCountList.get(attribute);
  }

  /**
   * Returns the root node of the tree.
   */
  public FPNode root() {
    return root;
  }

  /**
   * Adds an itemset with the given occurrance count.
   */
  public void accumulate(IntArrayList argItems, long count) {
    // boxed primitive used so we can use custom comparitor in sort
    List<Integer> items = Lists.newArrayList();
    for (int i = 0; i < argItems.size(); i++) {
      items.add(argItems.get(i));
    }
    Collections.sort(items, attrComparator);
    
    FPNode currNode = root;
    for (Integer item : items) {
      long attrCount = 0;
      if (item < attrCountList.size()) {
        attrCount = attrCountList.get(item);
      }
      if (attrCount < minSupport) {
        continue;
      }

      FPNode next = currNode.child(item);
      if (next == null) {
        next = new FPNode(currNode, item, count);
        currNode.addChild(next);
        List<FPNode> nodeList = attrNodeLists.get(item);
        if (nodeList == null) {
          nodeList = Lists.newArrayList();
          attrNodeLists.put(item, nodeList);
        }
        nodeList.add(next);
      } else {
        next.accumulate(count);
      }
      currNode = next;
    }
  } 

  /**
   * Adds an itemset with the given occurrance count.
   */
  public void accumulate(List<Integer> argItems, long count) {
    List<Integer> items = Lists.newArrayList();
    items.addAll(argItems);
    Collections.sort(items, attrComparator);
    
    FPNode currNode = root;
    for (Integer item : items) {
      long attrCount = attrCountList.get(item);
      if (attrCount < minSupport) {
        continue;
      }

      FPNode next = currNode.child(item);
      if (next == null) {
        next = new FPNode(currNode, item, count);
        currNode.addChild(next);
        List<FPNode> nodeList = attrNodeLists.get(item);
        if (nodeList == null) {
          nodeList = Lists.newArrayList();
          attrNodeLists.put(item, nodeList);
        }
        nodeList.add(next);
      } else {
        next.accumulate(count);
      }
      currNode = next;
    }
  }

  /**
   * Returns an Iterable over the attributes in the tree, sorted by
   * frequency (high to low).
   */
  public Iterable<Integer> attrIterableRev() {
    List<Integer> attrs = Lists.newArrayList();
    for (int i = 0; i < attrCountList.size(); i++) {
      if (attrCountList.get(i) > 0) {
        attrs.add(i);
      }
    }
    Collections.sort(attrs, Collections.reverseOrder(attrComparator));
    return attrs;
  }

  /**
   * Returns a conditional FP tree based on the targetAttr, containing
   * only items that are more frequent.
   */
  public FPTree createMoreFreqConditionalTree(int targetAttr) {
    LongArrayList counts = new LongArrayList();
    List<FPNode> nodeList = attrNodeLists.get(targetAttr);

    for (FPNode currNode : nodeList) {
      long pathCount = currNode.count();
      while (currNode != root) {
        int currAttr = currNode.attribute();
        if (counts.size() <= currAttr) {
          counts.setSize(currAttr + 1);
        }
        long count = counts.get(currAttr);
        counts.set(currNode.attribute(), count + pathCount);
        currNode = currNode.parent();
      }
    }
    if (counts.get(targetAttr) != attrCountList.get(targetAttr)) {
      throw new IllegalStateException("mismatched counts for targetAttr="
                                          + targetAttr + ", (" + counts.get(targetAttr)
                                          + " != " + attrCountList.get(targetAttr) + "); "
                                          + "thisTree=" + this + '\n');
    }
    counts.set(targetAttr, 0L);

    FPTree toRet = new FPTree(counts, minSupport);
    IntArrayList attrLst = new IntArrayList();
    for (FPNode currNode : attrNodeLists.get(targetAttr)) {
      long count = currNode.count();
      attrLst.clear();
      while (currNode != root) {
        if (currNode.count() < count) {
          throw new IllegalStateException();
        }
        attrLst.add(currNode.attribute());
        currNode = currNode.parent();
      }

      toRet.accumulate(attrLst, count);      
    }    
    return toRet;
  }

  // biggest count or smallest attr number goes first
  private class AttrComparator implements Comparator<Integer> {
    @Override
    public int compare(Integer a, Integer b) {

      long aCnt = 0;
      if (a < attrCountList.size()) {
        aCnt = attrCountList.get(a);
      }
      long bCnt = 0;
      if (b < attrCountList.size()) {
        bCnt = attrCountList.get(b);
      }
      if (aCnt == bCnt) {
        return a - b;
      }
      return (bCnt - aCnt) < 0 ? -1 : 1;
    }
  }

  /**
   *  Return a pair of trees that result from separating a common prefix
   *  (if one exists) from the lower portion of this tree.
   */
  public Pair<FPTree, FPTree> splitSinglePrefix() {
    if (root.numChildren() != 1) {
      return new Pair<FPTree, FPTree>(null, this);
    }
    LongArrayList pAttrCountList = new LongArrayList();
    LongArrayList qAttrCountList = attrCountList.copy();

    FPNode currNode = root;
    while (currNode.numChildren() == 1) {
      currNode = currNode.children().iterator().next();
      if (pAttrCountList.size() <= currNode.attribute()) {
        pAttrCountList.setSize(currNode.attribute() + 1);
      }
      pAttrCountList.set(currNode.attribute(), currNode.count());
      qAttrCountList.set(currNode.attribute(), 0);
    }

    FPTree pTree = new FPTree(pAttrCountList, minSupport);
    FPTree qTree = new FPTree(qAttrCountList, minSupport);
    recursivelyAddPrefixPats(pTree, qTree, root, null);

    return new Pair<FPTree, FPTree>(pTree, qTree);
  }

  private long recursivelyAddPrefixPats(FPTree pTree, FPTree qTree, FPNode node,
                                        IntArrayList items) {
    long count = node.count();
    int attribute = node.attribute();
    if (items == null) {
      // at root
      if (node != root) {
        throw new IllegalStateException();
      }
      items = new IntArrayList();
    } else {
      items.add(attribute);
    }
    long added = 0;
    for (FPNode child : node.children()) {
      added += recursivelyAddPrefixPats(pTree, qTree, child, items);
    }
    if (added < count) {
      long toAdd = count - added;
      pTree.accumulate(items, toAdd);
      qTree.accumulate(items, toAdd);
      added += toAdd;
    }
    if (node != root) {
      int lastIdx = items.size() - 1;
      if (items.get(lastIdx) != attribute) {
        throw new IllegalStateException();
      }
      items.remove(lastIdx);
    }
    return added;
  }

  private static void toStringHelper(StringBuilder sb, FPNode currNode, String prefix) {
    if (currNode.numChildren() == 0) {
      sb.append(prefix).append("-{attr:").append(currNode.attribute())
        .append(", cnt:").append(currNode.count()).append("}\n");
    } else {
      StringBuilder newPre = new StringBuilder(prefix);
      newPre.append("-{attr:").append(currNode.attribute())
        .append(", cnt:").append(currNode.count()).append('}');
      StringBuilder fakePre = new StringBuilder();
      while (fakePre.length() < newPre.length()) {
        fakePre.append(' ');
      }
      int i = 0;
      for (FPNode child : currNode.children()) {
        toStringHelper(sb, child, (i++ == 0 ? newPre : fakePre).toString() + '-' + i + "->");
      }
    }
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("[FPTree\n");
    toStringHelper(sb, root, "  ");
    sb.append(']');
    return sb.toString();
  }

}
