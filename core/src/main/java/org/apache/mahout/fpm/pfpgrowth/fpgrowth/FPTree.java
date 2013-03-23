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

import java.util.Arrays;
import java.util.Collection;
import java.util.TreeSet;

/**
 * The Frequent Pattern Tree datastructure used for mining patterns using
 * {@link FPGrowth} algorithm
 * 
 */
public class FPTree {
  
  public static final int ROOTNODEID = 0;
  private static final int DEFAULT_CHILDREN_INITIAL_SIZE = 2;
  private static final int DEFAULT_HEADER_TABLE_INITIAL_SIZE = 4;
  private static final int DEFAULT_INITIAL_SIZE = 8;
  private static final float GROWTH_RATE = 1.5f;
  private static final int HEADERTABLEBLOCKSIZE = 2;
  private static final int HT_LAST = 1;
  private static final int HT_NEXT = 0;
  
  private int[] attribute;
  private int[] childCount;
  private int[] conditional;
  private long[] headerTableAttributeCount;
  private int[] headerTableAttributes;
  private int headerTableCount;
  private int[] headerTableLookup;
  private int[][] headerTableProperties;
  private int[] next;
  private int[][] nodeChildren;
  private long[] nodeCount;
  private int nodes;
  private int[] parent;
  private boolean singlePath;
  private final Collection<Integer> sortedSet = new TreeSet<Integer>();
  
  public FPTree() {
    this(DEFAULT_INITIAL_SIZE);
  }
  
  public FPTree(int size) {
    if (size < DEFAULT_INITIAL_SIZE) {
      size = DEFAULT_INITIAL_SIZE;
    }
    
    parent = new int[size];
    next = new int[size];
    childCount = new int[size];
    attribute = new int[size];
    nodeCount = new long[size];
    
    nodeChildren = new int[size][];
    conditional = new int[size];
    
    headerTableAttributes = new int[DEFAULT_HEADER_TABLE_INITIAL_SIZE];
    headerTableAttributeCount = new long[DEFAULT_HEADER_TABLE_INITIAL_SIZE];
    headerTableLookup = new int[DEFAULT_HEADER_TABLE_INITIAL_SIZE];
    Arrays.fill(headerTableLookup, -1);
    headerTableProperties = new int[DEFAULT_HEADER_TABLE_INITIAL_SIZE][];
    
    singlePath = true;
    createRootNode();
  }
  
  public final void addChild(int parentNodeId, int childnodeId) {
    int length = childCount[parentNodeId];
    if (length >= nodeChildren[parentNodeId].length) {
      resizeChildren(parentNodeId);
    }
    nodeChildren[parentNodeId][length++] = childnodeId;
    childCount[parentNodeId] = length;
    
    if (length > 1 && singlePath) {
      singlePath = false;
    }
  }
  
  public final void addCount(int nodeId, long count) {
    if (nodeId < nodes) {
      this.nodeCount[nodeId] += count;
    }
  }
  
  public final void addHeaderCount(int attributeValue, long count) {
    int index = getHeaderIndex(attributeValue);
    headerTableAttributeCount[index] += count;
  }
  
  public final void addHeaderNext(int attributeValue, int nodeId) {
    int index = getHeaderIndex(attributeValue);
    if (headerTableProperties[index][HT_NEXT] == -1) {
      headerTableProperties[index][HT_NEXT] = nodeId;
      headerTableProperties[index][HT_LAST] = nodeId;
    } else {
      setNext(headerTableProperties[index][HT_LAST], nodeId);
      headerTableProperties[index][HT_LAST] = nodeId;
    }
  }
  
  public final int attribute(int nodeId) {
    return this.attribute[nodeId];
  }
  
  public final int childAtIndex(int nodeId, int index) {
    if (childCount[nodeId] < index) {
      return -1;
    }
    return nodeChildren[nodeId][index];
  }
  
  public final int childCount(int nodeId) {
    return childCount[nodeId];
  }
  
  public final int childWithAttribute(int nodeId, int childAttribute) {
    int length = childCount[nodeId];
    for (int i = 0; i < length; i++) {
      if (attribute[nodeChildren[nodeId][i]] == childAttribute) {
        return nodeChildren[nodeId][i];
      }
    }
    return -1;
  }
  
  public final void clear() {
    nodes = 0;
    headerTableCount = 0;
    singlePath = true;
    Arrays.fill(headerTableLookup, -1);
    sortedSet.clear();
    createRootNode();
  }
  
  public final void clearConditional() {
    for (int i = nodes - 1; i >= 0; i--) {
      conditional[i] = 0;
    }
  }
  
  public final int conditional(int nodeId) {
    return this.conditional[nodeId];
  }
  
  public final long count(int nodeId) {
    return nodeCount[nodeId];
  }
  
  public final int createConditionalNode(int attributeValue, long count) {
    if (nodes >= this.attribute.length) {
      resize();
    }
    childCount[nodes] = 0;
    next[nodes] = -1;
    parent[nodes] = -1;
    conditional[nodes] = 0;
    this.attribute[nodes] = attributeValue;
    nodeCount[nodes] = count;
    
    if (nodeChildren[nodes] == null) {
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];
    }
    
    return nodes++;
  }
  
  public final int createNode(int parentNodeId, int attributeValue, long count) {
    if (nodes >= this.attribute.length) {
      resize();
    }
    
    childCount[nodes] = 0;
    next[nodes] = -1;
    parent[nodes] = parentNodeId;
    this.attribute[nodes] = attributeValue;
    nodeCount[nodes] = count;
    
    conditional[nodes] = 0;
    if (nodeChildren[nodes] == null) {
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];
    }
    
    int childNodeId = nodes++;
    addChild(parentNodeId, childNodeId);
    addHeaderNext(attributeValue, childNodeId);
    return childNodeId;
  }
  
  public final void createRootNode() {
    childCount[nodes] = 0;
    next[nodes] = -1;
    parent[nodes] = 0;
    attribute[nodes] = -1;
    nodeCount[nodes] = 0;
    if (nodeChildren[nodes] == null) {
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];
    }
    nodes++;
  }
  
  public final int getAttributeAtIndex(int index) {
    return headerTableAttributes[index];
  }
  
  public final int getHeaderNext(int attributeValue) {
    int index = getHeaderIndex(attributeValue);
    return headerTableProperties[index][HT_NEXT];
  }
  
  public final long getHeaderSupportCount(int attributeValue) {
    int index = getHeaderIndex(attributeValue);
    return headerTableAttributeCount[index];
  }
  
  public final int[] getHeaderTableAttributes() {
    int[] attributes = new int[headerTableCount];
    System.arraycopy(headerTableAttributes, 0, attributes, 0, headerTableCount);
    return attributes;
  }
  
  public final int getHeaderTableCount() {
    return headerTableCount;
  }
  
  public final boolean isEmpty() {
    return nodes <= 1;
  }
  
  public final int next(int nodeId) {
    return next[nodeId];
  }
  
  public final int parent(int nodeId) {
    return parent[nodeId];
  }
  
  public final void removeHeaderNext(int attributeValue) {
    int index = getHeaderIndex(attributeValue);
    headerTableProperties[index][HT_NEXT] = -1;
  }
  
  public final void reorderHeaderTable() {
    // Arrays.sort(headerTableAttributes, 0, headerTableCount);
    int i = 0;
    for (int attr : sortedSet) {
      headerTableAttributes[i++] = attr;
    }
  }
  
  public void replaceChild(int parentNodeId, int replacableNode, int childnodeId) {
    int max = childCount[parentNodeId];
    for (int i = 0; i < max; i++) {
      if (nodeChildren[parentNodeId][i] == replacableNode) {
        nodeChildren[parentNodeId][i] = childnodeId;
        parent[childnodeId] = parentNodeId;
      }
    }
  }
  
  public final void setConditional(int nodeId, int conditionalNode) {
    if (nodeId < nodes) {
      this.conditional[nodeId] = conditionalNode;
    }
  }
  
  public final void setNext(int nodeId, int nextNode) {
    if (nodeId < nodes) {
      this.next[nodeId] = nextNode;
    }
  }
  
  public final void setParent(int nodeId, int parentNode) {
    if (nodeId < nodes) {
      this.parent[nodeId] = parentNode;
      
      int length = childCount[parentNode];
      if (length >= nodeChildren[parentNode].length) {
        resizeChildren(parentNode);
      }
      nodeChildren[parentNode][length++] = nodeId;
      childCount[parentNode] = length;
    }
  }
  
  public final void setSinglePath(boolean bit) {
    singlePath = bit;
  }
  
  public final boolean singlePath() {
    return singlePath;
  }
  
  private int getHeaderIndex(int attributeValue) {
    if (attributeValue >= headerTableLookup.length) {
      resizeHeaderLookup(attributeValue);
    }
    int index = headerTableLookup[attributeValue];
    if (index == -1) { // if attribute didnt exist;
      if (headerTableCount >= headerTableAttributes.length) {
        resizeHeaderTable();
      }
      headerTableAttributes[headerTableCount] = attributeValue;
      if (headerTableProperties[headerTableCount] == null) {
        headerTableProperties[headerTableCount] = new int[HEADERTABLEBLOCKSIZE];
      }
      headerTableAttributeCount[headerTableCount] = 0;
      headerTableProperties[headerTableCount][HT_NEXT] = -1;
      headerTableProperties[headerTableCount][HT_LAST] = -1;
      index = headerTableCount++;
      headerTableLookup[attributeValue] = index;
      sortedSet.add(attributeValue);
    }
    return index;
  }
  
  private void resize() {
    int size = (int) (GROWTH_RATE * nodes);
    if (size < DEFAULT_INITIAL_SIZE) {
      size = DEFAULT_INITIAL_SIZE;
    }
    
    int[] oldChildCount = childCount;
    int[] oldAttribute = attribute;
    long[] oldnodeCount = nodeCount;
    int[] oldParent = parent;
    int[] oldNext = next;
    int[][] oldNodeChildren = nodeChildren;
    int[] oldConditional = conditional;
    
    childCount = new int[size];
    attribute = new int[size];
    nodeCount = new long[size];
    parent = new int[size];
    next = new int[size];
    
    nodeChildren = new int[size][];
    conditional = new int[size];
    
    System.arraycopy(oldChildCount, 0, this.childCount, 0, nodes);
    System.arraycopy(oldAttribute, 0, this.attribute, 0, nodes);
    System.arraycopy(oldnodeCount, 0, this.nodeCount, 0, nodes);
    System.arraycopy(oldParent, 0, this.parent, 0, nodes);
    System.arraycopy(oldNext, 0, this.next, 0, nodes);
    System.arraycopy(oldNodeChildren, 0, this.nodeChildren, 0, nodes);
    System.arraycopy(oldConditional, 0, this.conditional, 0, nodes);
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
  
  private void resizeHeaderLookup(int attributeValue) {
    int size = (int) (attributeValue * GROWTH_RATE);
    int[] oldLookup = headerTableLookup;
    headerTableLookup = new int[size];
    Arrays.fill(headerTableLookup, oldLookup.length, size, -1);
    System.arraycopy(oldLookup, 0, this.headerTableLookup, 0, oldLookup.length);
  }
  
  private void resizeHeaderTable() {
    int size = (int) (GROWTH_RATE * headerTableCount);
    if (size < DEFAULT_HEADER_TABLE_INITIAL_SIZE) {
      size = DEFAULT_HEADER_TABLE_INITIAL_SIZE;
    }
    
    int[] oldAttributes = headerTableAttributes;
    long[] oldAttributeCount = headerTableAttributeCount;
    int[][] oldProperties = headerTableProperties;
    headerTableAttributes = new int[size];
    headerTableAttributeCount = new long[size];
    headerTableProperties = new int[size][];
    System.arraycopy(oldAttributes, 0, this.headerTableAttributes, 0,
      headerTableCount);
    System.arraycopy(oldAttributeCount, 0, this.headerTableAttributeCount, 0,
      headerTableCount);
    System.arraycopy(oldProperties, 0, this.headerTableProperties, 0,
      headerTableCount);
  }

  private void toStringHelper(StringBuilder sb, int currNode, String prefix) {
    if (childCount[currNode] == 0) {
      sb.append(prefix).append("-{attr:").append(attribute[currNode])
        .append(", id: ").append(currNode)
        .append(", cnt:").append(nodeCount[currNode]).append("}\n");
    } else {
      StringBuilder newPre = new StringBuilder(prefix);
      newPre.append("-{attr:").append(attribute[currNode])
        .append(", id: ").append(currNode)
        .append(", cnt:").append(nodeCount[currNode]).append('}');
      StringBuilder fakePre = new StringBuilder();
      while (fakePre.length() < newPre.length()) {
        fakePre.append(' ');
      }
      for (int i = 0; i < childCount[currNode]; i++) {
        toStringHelper(sb, nodeChildren[currNode][i], (i == 0 ? newPre : fakePre).toString() + '-' + i + "->");
      }
    }
  }
  
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("[FPTree\n");
    toStringHelper(sb, 0, "  ");
    sb.append("\n]\n");
    return sb.toString();
  }

}
