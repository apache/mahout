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

package org.apache.mahout.classifier.df;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataUtils;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.node.Node;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Represents a forest of decision trees.
 */
public class DecisionForest implements Writable {
  
  private final List<Node> trees;
  
  private DecisionForest() {
    trees = Lists.newArrayList();
  }
  
  public DecisionForest(List<Node> trees) {
    Preconditions.checkArgument(trees != null && !trees.isEmpty(), "trees argument must not be null or empty");

    this.trees = trees;
  }
  
  List<Node> getTrees() {
    return trees;
  }

  /**
   * Classifies the data and calls callback for each classification
   */
  public void classify(Data data, double[][] predictions) {
    Preconditions.checkArgument(data.size() == predictions.length, "predictions.length must be equal to data.size()");

    if (data.isEmpty()) {
      return; // nothing to classify
    }

    int treeId = 0;
    for (Node tree : trees) {
      for (int index = 0; index < data.size(); index++) {
        if (predictions[index] == null) {
          predictions[index] = new double[trees.size()];
        }
        predictions[index][treeId] = tree.classify(data.get(index));
      }
      treeId++;
    }
  }
  
  /**
   * predicts the label for the instance
   * 
   * @param rng
   *          Random number generator, used to break ties randomly
   * @return NaN if the label cannot be predicted
   */
  public double classify(Dataset dataset, Random rng, Instance instance) {
    if (dataset.isNumerical(dataset.getLabelId())) {
      double sum = 0;
      int cnt = 0;
      for (Node tree : trees) {
        double prediction = tree.classify(instance);
        if (!Double.isNaN(prediction)) {
          sum += prediction;
          cnt++;
        }
      }

      if (cnt > 0) {
        return sum / cnt;
      } else {
        return Double.NaN;
      }
    } else {
      int[] predictions = new int[dataset.nblabels()];
      for (Node tree : trees) {
        double prediction = tree.classify(instance);
        if (!Double.isNaN(prediction)) {
          predictions[(int) prediction]++;
        }
      }

      if (DataUtils.sum(predictions) == 0) {
        return Double.NaN; // no prediction available
      }

      return DataUtils.maxindex(rng, predictions);
    }
  }
  
  /**
   * @return Mean number of nodes per tree
   */
  public long meanNbNodes() {
    long sum = 0;
    
    for (Node tree : trees) {
      sum += tree.nbNodes();
    }
    
    return sum / trees.size();
  }
  
  /**
   * @return Total number of nodes in all the trees
   */
  public long nbNodes() {
    long sum = 0;
    
    for (Node tree : trees) {
      sum += tree.nbNodes();
    }
    
    return sum;
  }
  
  /**
   * @return Mean maximum depth per tree
   */
  public long meanMaxDepth() {
    long sum = 0;
    
    for (Node tree : trees) {
      sum += tree.maxDepth();
    }
    
    return sum / trees.size();
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof DecisionForest)) {
      return false;
    }
    
    DecisionForest rf = (DecisionForest) obj;
    
    return trees.size() == rf.getTrees().size() && trees.containsAll(rf.getTrees());
  }
  
  @Override
  public int hashCode() {
    return trees.hashCode();
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeInt(trees.size());
    for (Node tree : trees) {
      tree.write(dataOutput);
    }
  }

  /**
   * Reads the trees from the input and adds them to the existing trees
   */
  @Override
  public void readFields(DataInput dataInput) throws IOException {
    int size = dataInput.readInt();
    for (int i = 0; i < size; i++) {
      trees.add(Node.read(dataInput));
    }
  }

  private static DecisionForest read(DataInput dataInput) throws IOException {
    DecisionForest forest = new DecisionForest();
    forest.readFields(dataInput);
    return forest;
  }

  /**
   * Load the forest from a single file or a directory of files
   * @throws java.io.IOException
   */
  public static DecisionForest load(Configuration conf, Path forestPath) throws IOException {
    FileSystem fs = forestPath.getFileSystem(conf);
    Path[] files;
    if (fs.getFileStatus(forestPath).isDir()) {
      files = DFUtils.listOutputFiles(fs, forestPath);
    } else {
      files = new Path[]{forestPath};
    }

    DecisionForest forest = null;
    for (Path path : files) {
      FSDataInputStream dataInput = new FSDataInputStream(fs.open(path));
      try {
        if (forest == null) {
          forest = read(dataInput);
        } else {
          forest.readFields(dataInput);
        }
      } finally {
        Closeables.close(dataInput, true);
      }
    }

    return forest;
    
  }

}
