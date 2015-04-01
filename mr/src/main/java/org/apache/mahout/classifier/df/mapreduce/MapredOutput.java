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

package org.apache.mahout.classifier.df.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.node.Node;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/**
 * Used by various implementation to return the results of a build.<br>
 * Contains a grown tree and and its oob predictions.
 */
public class MapredOutput implements Writable, Cloneable {

  private Node tree;

  private int[] predictions;

  public MapredOutput() {
  }

  public MapredOutput(Node tree, int[] predictions) {
    this.tree = tree;
    this.predictions = predictions;
  }

  public MapredOutput(Node tree) {
    this(tree, null);
  }

  public Node getTree() {
    return tree;
  }

  int[] getPredictions() {
    return predictions;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    boolean readTree = in.readBoolean();
    if (readTree) {
      tree = Node.read(in);
    }

    boolean readPredictions = in.readBoolean();
    if (readPredictions) {
      predictions = DFUtils.readIntArray(in);
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(tree != null);
    if (tree != null) {
      tree.write(out);
    }

    out.writeBoolean(predictions != null);
    if (predictions != null) {
      DFUtils.writeArray(out, predictions);
    }
  }

  @Override
  public MapredOutput clone() {
    return new MapredOutput(tree, predictions);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof MapredOutput)) {
      return false;
    }

    MapredOutput mo = (MapredOutput) obj;

    return ((tree == null && mo.getTree() == null) || (tree != null && tree.equals(mo.getTree())))
        && Arrays.equals(predictions, mo.getPredictions());
  }

  @Override
  public int hashCode() {
    int hashCode = tree == null ? 1 : tree.hashCode();
    for (int prediction : predictions) {
      hashCode = 31 * hashCode + prediction;
    }
    return hashCode;
  }

  @Override
  public String toString() {
    return "{" + tree + " | " + Arrays.toString(predictions) + '}';
  }

}
