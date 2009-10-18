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

public class FPTree {

  final public static int DEFAULT_CHILDREN_INITIAL_SIZE = 2;

  final public static int DEFAULT_HEADER_TABLE_INITIAL_SIZE = 4;

  final public static int DEFAULT_INITIAL_SIZE = 8;

  final public float GROWTH_RATE = 1.5f;

  final public static int ROOTNODEID = 0;

  final private static int HEADERTABLEBLOCKSIZE = 2;

  final private static int HT_LAST = 1;

  final private static int HT_NEXT = 0;

  private int[] attribute;

  private int[] childCount;

  private int[] conditional;

  private int[] headerTableAttributes;

  private long[] headerTableAttributeCount;

  private int headerTableCount = 0;

  private int[] headerTableLookup;

  private int[][] headerTableProperties;

  private int[] next;

  private int[][] nodeChildren;

  private long[] nodeCount;

  private int nodes = 0;

  private int[] parent;

  private boolean singlePath;

  public FPTree() {
    this(DEFAULT_INITIAL_SIZE, DEFAULT_HEADER_TABLE_INITIAL_SIZE);
  }

  public FPTree(int size) {
    this(size, DEFAULT_HEADER_TABLE_INITIAL_SIZE);
  }

  public FPTree(int size, int headersize) {
    if (size < DEFAULT_INITIAL_SIZE)
      size = DEFAULT_INITIAL_SIZE;

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

  final public void addChild(int parentNodeId, int childnodeId) {
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

  final public boolean addCount(int nodeId, long nextNodeCount) {
    if (nodeId < nodes) {
      this.nodeCount[nodeId] += nextNodeCount;
      return true;
    }
    return false;
  }

  final public void addHeaderCount(int attribute, long count) {
    int index = getHeaderIndex(attribute);
    headerTableAttributeCount[index] += count;
  }

  final public void addHeaderNext(int attribute, int nodeId) {
    int index = getHeaderIndex(attribute);
    if (headerTableProperties[index][HT_NEXT] == -1) {
      headerTableProperties[index][HT_NEXT] = nodeId;
      headerTableProperties[index][HT_LAST] = nodeId;
    } else {
      setNext(headerTableProperties[index][HT_LAST], nodeId);
      headerTableProperties[index][HT_LAST] = nodeId;
    }
  }

  final public int attribute(int nodeId) {
    return this.attribute[nodeId];
  }

  final public int childAtIndex(int nodeId, int index) {
    if (childCount[nodeId] < index) {
      return -1;
    }
    return nodeChildren[nodeId][index];
  }

  final public int childCount(int nodeId) {
    return childCount[nodeId];
  }

  final public int childWithAttribute(int nodeId, int childAttribute) {
    int length = childCount[nodeId];
    for (int i = 0; i < length; i++) {
      if (attribute[nodeChildren[nodeId][i]] == childAttribute)
        return nodeChildren[nodeId][i];
    }
    return -1;
  }

  final public void clear() {
    nodes = 0;
    headerTableCount = 0;
    Arrays.fill(headerTableLookup, -1);
    createRootNode();
  }

  final public void clearConditional() {
    for (int i = nodes - 1; i >= 0; i--)
      conditional[i] = 0;
  }

  final public int conditional(int nodeId) {
    return this.conditional[nodeId];
  }

  final public long count(int nodeId) {
    return nodeCount[nodeId];
  }

  final public int createConditionalNode(int attribute, long count) {
    if (nodes >= this.attribute.length) {
      resize();
    }
    childCount[nodes] = 0;
    next[nodes] = -1;
    parent[nodes] = -1;
    conditional[nodes] = 0;
    this.attribute[nodes] = attribute;
    nodeCount[nodes] = count;

    if (nodeChildren[nodes] == null)
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];

    int childNodeId = nodes++;
    return childNodeId;
  }

  final public int createNode(int parentNodeId, int attribute, long count) {
    if (nodes >= this.attribute.length) {
      resize();
    }

    childCount[nodes] = 0;
    next[nodes] = -1;
    parent[nodes] = parentNodeId;
    this.attribute[nodes] = attribute;
    nodeCount[nodes] = count;

    conditional[nodes] = 0;
    if (nodeChildren[nodes] == null)
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];

    int childNodeId = nodes++;
    addChild(parentNodeId, childNodeId);
    addHeaderNext(attribute, childNodeId);
    return childNodeId;
  }

  final public int createRootNode() {
    childCount[nodes] = 0;
    next[nodes] = -1;
    parent[nodes] = 0;
    attribute[nodes] = -1;
    nodeCount[nodes] = 0;
    if (nodeChildren[nodes] == null)
      nodeChildren[nodes] = new int[DEFAULT_CHILDREN_INITIAL_SIZE];
    int childNodeId = nodes++;
    return childNodeId;
  }

  final public int getAttributeAtIndex(int index) {
    return headerTableAttributes[index];
  }

  final public int getHeaderNext(int attribute) {
    int index = getHeaderIndex(attribute);
    return headerTableProperties[index][HT_NEXT];
  }

  final public long getHeaderSupportCount(int attribute) {
    int index = getHeaderIndex(attribute);
    return headerTableAttributeCount[index];
  }

  final public int[] getHeaderTableAttributes() {
    int[] attributes = new int[headerTableCount];
    System.arraycopy(headerTableAttributes, 0, attributes, 0, headerTableCount);
    return attributes;
  }

  final public int getHeaderTableCount() {
    return headerTableCount;
  }

  final public int next(int nodeId) {
    return next[nodeId];
  }

  final public boolean isEmpty() {
    return nodes <= 1;
  }

  final public int parent(int nodeId) {
    return parent[nodeId];
  }

  final public void reorderHeaderTable() {
    Arrays.sort(headerTableAttributes, 0, headerTableCount);
  }

  final public boolean setConditional(int nodeId, int conditional) {
    if (nodeId < nodes) {
      this.conditional[nodeId] = conditional;
      return true;
    }
    return false;
  }

  final public boolean setNext(int nodeId, int next) {
    if (nodeId < nodes) {
      this.next[nodeId] = next;
      return true;
    }
    return false;
  }

  final public boolean setParent(int nodeId, int parent) {
    if (nodeId < nodes) {
      this.parent[nodeId] = parent;

      int length = childCount[parent];
      if (length >= nodeChildren[parent].length) {
        resizeChildren(parent);
      }
      nodeChildren[parent][length++] = nodeId;
      childCount[parent] = length;
      return true;
    }
    return false;
  }

  final public void setSinglePath(boolean bit) {
    singlePath = bit;
  }

  final public boolean singlePath() {
    return singlePath;
  }

  final private int getHeaderIndex(int attribute) {
    if (attribute >= headerTableLookup.length)
      resizeHeaderLookup(attribute);
    int index = headerTableLookup[attribute];
    if (index == -1) { // if attribute didnt exist;
      if (headerTableCount >= headerTableAttributes.length)
        resizeHeaderTable();
      headerTableAttributes[headerTableCount] = attribute;
      if (headerTableProperties[headerTableCount] == null)
        headerTableProperties[headerTableCount] = new int[HEADERTABLEBLOCKSIZE];
      headerTableAttributeCount[headerTableCount] = 0;
      headerTableProperties[headerTableCount][HT_NEXT] = -1;
      headerTableProperties[headerTableCount][HT_LAST] = -1;
      index = headerTableCount++;
      headerTableLookup[attribute] = index;
    }
    return index;
  }

  final private void resize() {
    int size = (int) (GROWTH_RATE * nodes);
    if (size < DEFAULT_INITIAL_SIZE)
      size = DEFAULT_INITIAL_SIZE;

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

  final private void resizeChildren(int nodeId) {
    int length = childCount[nodeId];
    int size = (int) (GROWTH_RATE * (length));
    if (size < DEFAULT_CHILDREN_INITIAL_SIZE)
      size = DEFAULT_CHILDREN_INITIAL_SIZE;
    int[] oldNodeChildren = nodeChildren[nodeId];
    nodeChildren[nodeId] = new int[size];
    System.arraycopy(oldNodeChildren, 0, this.nodeChildren[nodeId], 0, length);
  }

  final private void resizeHeaderLookup(int attribute) {
    int size = (int) (attribute * GROWTH_RATE);
    int[] oldLookup = headerTableLookup;
    headerTableLookup = new int[size];
    Arrays.fill(headerTableLookup, oldLookup.length, size, -1);
    System.arraycopy(oldLookup, 0, this.headerTableLookup, 0, oldLookup.length);
  }

  final private void resizeHeaderTable() {
    int size = (int) (GROWTH_RATE * (headerTableCount));
    if (size < DEFAULT_HEADER_TABLE_INITIAL_SIZE)
      size = DEFAULT_HEADER_TABLE_INITIAL_SIZE;

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
}
