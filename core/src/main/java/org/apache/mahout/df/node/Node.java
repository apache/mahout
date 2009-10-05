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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.df.data.Instance;

/**
 * Represents an abstract node of a decision tree
 */
public abstract class Node implements Writable {

  //protected static final String DELIMITER = ",";

  protected enum NODE_TYPE {
    MOCKLEAF, LEAF, NUMERICAL, CATEGORICAL
  }

  /**
   * predicts the label for the instance
   * 
   * @param instance
   * @return -1 if the label cannot be predicted
   */
  public abstract int classify(Instance instance);

  /**
   * returns the total number of nodes of the tree
   * 
   * @return
   */
  public abstract long nbNodes();

  /**
   * returns the maximum depth of the tree
   * 
   * @return
   */
  public abstract long maxDepth();

  /**
   * converts the node implementation into an int code
   * 
   * @return
   */
  private int node2Type() {
    if (this instanceof MockLeaf) {
      return NODE_TYPE.MOCKLEAF.ordinal();
    } else if (this instanceof Leaf) {
      return NODE_TYPE.LEAF.ordinal();
    } else if (this instanceof NumericalNode) {
      return NODE_TYPE.NUMERICAL.ordinal();
    } else if (this instanceof CategoricalNode) {
      return NODE_TYPE.CATEGORICAL.ordinal();
    } else {
      throw new IllegalStateException(
          "This implementation is not currently supported");
    }
  }

  public static Node read(DataInput in) throws IOException {
    NODE_TYPE type = NODE_TYPE.values()[in.readInt()];
    Node node;

    switch (type) {
      case MOCKLEAF:
        node = new MockLeaf();
        break;
      case LEAF:
        node = new Leaf();
        break;
      case NUMERICAL:
        node = new NumericalNode();
        break;
      case CATEGORICAL:
        node = new CategoricalNode();
        break;
      default:
        throw new IllegalStateException(
            "This implementation is not currently supported");
    }

    node.readFields(in);

    return node;
  }

  @Override
  public final String toString() {
    return node2Type() + ":" + getString() + ';';
  }

  protected abstract String getString();

  @Override
  public final void write(DataOutput out) throws IOException {
    out.writeInt(node2Type());
    writeNode(out);
  }

  protected abstract void writeNode(DataOutput out) throws IOException;

}
