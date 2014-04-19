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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.mahout.math.list.IntArrayList;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.lang3.mutable.MutableLong;
import org.apache.hadoop.io.VIntWritable;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A compact representation of transactions modeled on the lines to
 * {@link org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPTree} This reduces plenty of space and speeds up
 * Map/Reduce of {@link PFPGrowth} algorithm by reducing data size passed from the Mapper to the reducer where
 * {@link org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth} mining is done
 */
public final class TransactionTree implements Writable, Iterable<Pair<IntArrayList,Long>> {

  private static final Logger log = LoggerFactory.getLogger(TransactionTree.class);

  private static final int DEFAULT_CHILDREN_INITIAL_SIZE = 2;
  private static final int DEFAULT_INITIAL_SIZE = 8;
  private static final float GROWTH_RATE = 1.5f;
  private static final int ROOTNODEID = 0;
  
  private int[] attribute;
  private int[] childCount;
  private int[][] nodeChildren;
  private long[] nodeCount;
  private int nodes;
  private boolean representedAsList;
  private List<Pair<IntArrayList,Long>> transactionSet;
  
  public TransactionTree() {
    this(DEFAULT_INITIAL_SIZE);
  }
  
  public TransactionTree(int size) {
    if (size < DEFAULT_INITIAL_SIZE) {
      size = DEFAULT_INITIAL_SIZE;
    }
    childCount = new int[size];
    attribute = new int[size];
    nodeCount = new long[size];
    nodeChildren = new int[size][];
    createRootNode();
    representedAsList = false;
  }

  public TransactionTree(IntArrayList items, Long support) {
    representedAsList = true;
    transactionSet = Lists.newArrayList();
    transactionSet.add(new Pair<IntArrayList,Long>(items, support));
  }
  
  public TransactionTree(List<Pair<IntArrayList,Long>> transactionSet) {
    representedAsList = true;
    this.transactionSet = transactionSet;
  }
  
  public void addChild(int parentNodeId, int childnodeId) {
    int length = childCount[parentNodeId];
    if (length >= nodeChildren[parentNodeId].length) {
      resizeChildren(parentNodeId);
    }
    nodeChildren[parentNodeId][length++] = childnodeId;
    childCount[parentNodeId] = length;
    
  }
  
  public void addCount(int nodeId, long nextNodeCount) {
    if (nodeId < nodes) {
      this.nodeCount[nodeId] += nextNodeCount;
    }
  }

  public int addPattern(IntArrayList myList, long addCount) {
    int temp = ROOTNODEID;
    int ret = 0;
    boolean addCountMode = true;
    for (int idx = 0; idx < myList.size(); idx++) {
      int attributeValue = myList.get(idx);
      int child;
      if (addCountMode) {
        child = childWithAttribute(temp, attributeValue);
        if (child == -1) {
          addCountMode = false;
        } else {
          addCount(child, addCount);
          temp = child;
        }
      }
      if (!addCountMode) {
        child = createNode(temp, attributeValue, addCount);
        temp = child;
        ret++;
      }
    }
    return ret;
  }
  
  public int attribute(int nodeId) {
    return this.attribute[nodeId];
  }
  
  public int childAtIndex(int nodeId, int index) {
    if (childCount[nodeId] < index) {
      return -1;
    }
    return nodeChildren[nodeId][index];
  }
  
  public int childCount() {
    int sum = 0;
    for (int i = 0; i < nodes; i++) {
      sum += childCount[i];
    }
    return sum;
  }
  
  public int childCount(int nodeId) {
    return childCount[nodeId];
  }
  
  public int childWithAttribute(int nodeId, int childAttribute) {
    int length = childCount[nodeId];
    for (int i = 0; i < length; i++) {
      if (attribute[nodeChildren[nodeId][i]] == childAttribute) {
        return nodeChildren[nodeId][i];
      }
    }
    return -1;
  }
  
  public long count(int nodeId) {
    return nodeCount[nodeId];
  }
  
  public Map<Integer,MutableLong> generateFList() {
    Map<Integer,MutableLong> frequencyList = Maps.newHashMap();
    for (Pair<IntArrayList, Long> p : this) {
      IntArrayList items = p.getFirst();
      for (int idx = 0; idx < items.size(); idx++) {
        if (!frequencyList.containsKey(items.get(idx))) {
          frequencyList.put(items.get(idx), new MutableLong(0));
        }
        frequencyList.get(items.get(idx)).add(p.getSecond());
      }
    }
    return frequencyList;
  }
  
  public TransactionTree getCompressedTree() {
    TransactionTree ctree = new TransactionTree();
    Iterator<Pair<IntArrayList,Long>> it = iterator();
    int node = 0;
    int size = 0;
    List<Pair<IntArrayList,Long>> compressedTransactionSet = Lists.newArrayList();
    while (it.hasNext()) {
      Pair<IntArrayList,Long> p = it.next();
      p.getFirst().sort();
      compressedTransactionSet.add(p);
      node += ctree.addPattern(p.getFirst(), p.getSecond());
      size += p.getFirst().size() + 2;
    }

    if (log.isDebugEnabled()) {
      log.debug("Nodes in UnCompressed Tree: {} ", nodes);
      log.debug("UnCompressed Tree Size: {}", (this.nodes * 4 * 4 + this.childCount() * 4) / 1000000.0);
      log.debug("Nodes in Compressed Tree: {} ", node);
      log.debug("Compressed Tree Size: {}", (node * 4 * 4 + ctree.childCount() * 4) / 1000000.0);
      log.debug("TransactionSet Size: {}", size * 4 / 1000000.0);
    }
    if (node * 4 * 4 + ctree.childCount() * 4 <= size * 4) {
      return ctree;
    } else {
      return new TransactionTree(compressedTransactionSet);
    }
  }
  
  @Override
  public Iterator<Pair<IntArrayList,Long>> iterator() {
    if (this.isTreeEmpty() && !representedAsList) {
      throw new IllegalStateException("This is a bug. Please report this to mahout-user list");
    } else if (representedAsList) {
      return transactionSet.iterator();
    } else {
      return new TransactionTreeIterator(this);
    }
  }
  
  public boolean isTreeEmpty() {
    return nodes <= 1;
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    representedAsList = in.readBoolean();
    
    VIntWritable vInt = new VIntWritable();
    VLongWritable vLong = new VLongWritable();
    
    if (representedAsList) {
      transactionSet = Lists.newArrayList();
      vInt.readFields(in);
      int numTransactions = vInt.get();
      for (int i = 0; i < numTransactions; i++) {
        vLong.readFields(in);
        Long support = vLong.get();
        
        vInt.readFields(in);
        int length = vInt.get();
        
        int[] items = new int[length];
        for (int j = 0; j < length; j++) {
          vInt.readFields(in);
          items[j] = vInt.get();
        }
        Pair<IntArrayList,Long> transaction = new Pair<IntArrayList,Long>(new IntArrayList(items), support);
        transactionSet.add(transaction);
      }
    } else {
      vInt.readFields(in);
      nodes = vInt.get();
      attribute = new int[nodes];
      nodeCount = new long[nodes];
      childCount = new int[nodes];
      nodeChildren = new int[nodes][];
      for (int i = 0; i < nodes; i++) {
        vInt.readFields(in);
        attribute[i] = vInt.get();
        vLong.readFields(in);
        nodeCount[i] = vLong.get();
        vInt.readFields(in);
        int childCountI = vInt.get();
        childCount[i] = childCountI;
        nodeChildren[i] = new int[childCountI];
        for (int j = 0; j < childCountI; j++) {
          vInt.readFields(in);
          nodeChildren[i][j] = vInt.get();
        }
      }
    }
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(representedAsList);
    VIntWritable vInt = new VIntWritable();
    VLongWritable vLong = new VLongWritable();
    if (representedAsList) {
      int transactionSetSize = transactionSet.size();
      vInt.set(transactionSetSize);
      vInt.write(out);
      for (Pair<IntArrayList, Long> transaction : transactionSet) {
        vLong.set(transaction.getSecond());
        vLong.write(out);

        vInt.set(transaction.getFirst().size());
        vInt.write(out);

        IntArrayList items = transaction.getFirst();
        for (int idx = 0; idx < items.size(); idx++) {
          int item = items.get(idx);
          vInt.set(item);
          vInt.write(out);
        }
      }
    } else {
      vInt.set(nodes);
      vInt.write(out);
      for (int i = 0; i < nodes; i++) {
        vInt.set(attribute[i]);
        vInt.write(out);
        vLong.set(nodeCount[i]);
        vLong.write(out);
        vInt.set(childCount[i]);
        vInt.write(out);
        int max = childCount[i];
        for (int j = 0; j < max; j++) {
          vInt.set(nodeChildren[i][j]);
          vInt.write(out);
        }
      }
    }
  }
  
  private int createNode(int parentNodeId, int attributeValue, long count) {
    if (nodes >= this.attribute.length) {
      resize();
    }
    
    childCount[nodes] = 0;
    this.attribute[nodes] = attributeValue;
    nodeCount[nodes] = count;
    if (nodeChildren[nodes] == null) {
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];
    }
    
    int childNodeId = nodes++;
    addChild(parentNodeId, childNodeId);
    return childNodeId;
  }
  
  private void createRootNode() {
    childCount[nodes] = 0;
    attribute[nodes] = -1;
    nodeCount[nodes] = 0;
    if (nodeChildren[nodes] == null) {
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];
    }
    nodes++;
  }
  
  private void resize() {
    int size = (int) (GROWTH_RATE * nodes);
    if (size < DEFAULT_INITIAL_SIZE) {
      size = DEFAULT_INITIAL_SIZE;
    }
    
    int[] oldChildCount = childCount;
    int[] oldAttribute = attribute;
    long[] oldnodeCount = nodeCount;
    int[][] oldNodeChildren = nodeChildren;
    
    childCount = new int[size];
    attribute = new int[size];
    nodeCount = new long[size];
    nodeChildren = new int[size][];
    
    System.arraycopy(oldChildCount, 0, this.childCount, 0, nodes);
    System.arraycopy(oldAttribute, 0, this.attribute, 0, nodes);
    System.arraycopy(oldnodeCount, 0, this.nodeCount, 0, nodes);
    System.arraycopy(oldNodeChildren, 0, this.nodeChildren, 0, nodes);
  }
  
  private void resizeChildren(int nodeId) {
    int length = childCount[nodeId];
    int size = (int) (GROWTH_RATE * length);
    if (size < DEFAULT_CHILDREN_INITIAL_SIZE) {
      size = DEFAULT_CHILDREN_INITIAL_SIZE;
    }
    int[] oldNodeChildren = nodeChildren[nodeId];
    nodeChildren[nodeId] = new int[size];
    System.arraycopy(oldNodeChildren, 0, this.nodeChildren[nodeId], 0, length);
  }
}
