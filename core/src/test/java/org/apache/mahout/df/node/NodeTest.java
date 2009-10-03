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

package org.apache.mahout.df.node;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;

import junit.framework.TestCase;

public class NodeTest extends TestCase {

  private Random rng;

  private ByteArrayOutputStream byteOutStream;
  private DataOutput out;
  
  @Override
  protected void setUp() throws Exception {
    rng = RandomUtils.getRandom();

    byteOutStream = new ByteArrayOutputStream();
    out = new DataOutputStream(byteOutStream);
  }

  /**
   * Test method for
   * {@link org.apache.mahout.df.node.Node#read(java.io.DataInput)}.
   */
  public void testReadTree() throws Exception {
    Node node1 = new CategoricalNode(rng.nextInt(), 
        new double[] { rng.nextDouble(), rng.nextDouble() }, 
        new Node[] { new Leaf(rng.nextInt()), new Leaf(rng.nextInt()) });
    Node node2 = new NumericalNode(rng.nextInt(), rng.nextDouble(), 
        new Leaf(rng.nextInt()), new Leaf(rng.nextInt()));
    
    Node root = new CategoricalNode(rng.nextInt(), 
        new double[] { rng.nextDouble(), rng.nextDouble(), rng.nextDouble() }, 
        new Node[] { node1, node2, new Leaf(rng.nextInt()) });

    // write the node to a DataOutput
    root.write(out);
    
    // read the node back
    assertEquals(root, readNode());
  }

  protected Node readNode() throws IOException {
    ByteArrayInputStream byteInStream = new ByteArrayInputStream(byteOutStream.toByteArray());
    DataInput in = new DataInputStream(byteInStream);
    return Node.read(in);
  }
  
  public void testReadLeaf() throws Exception {

    Leaf leaf = new Leaf(rng.nextInt());
    leaf.write(out);
    assertEquals(leaf, readNode());
  }

  public void testParseNumerical() throws Exception {

    NumericalNode node = new NumericalNode(rng.nextInt(), rng.nextDouble(), new Leaf(rng
        .nextInt()), new Leaf(rng.nextInt()));
    node.write(out);
    assertEquals(node, readNode());
  }

  public void testCategoricalNode() throws Exception {

    CategoricalNode node = new CategoricalNode(rng.nextInt(), new double[]{rng.nextDouble(),
        rng.nextDouble(), rng.nextDouble()}, new Node[]{
        new Leaf(rng.nextInt()), new Leaf(rng.nextInt()),
        new Leaf(rng.nextInt())});

    node.write(out);
    assertEquals(node, readNode());
  }
}
