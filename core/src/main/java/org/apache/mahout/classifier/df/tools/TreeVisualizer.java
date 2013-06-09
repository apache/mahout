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

package org.apache.mahout.classifier.df.tools;

import java.lang.reflect.Field;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.node.CategoricalNode;
import org.apache.mahout.classifier.df.node.Leaf;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.classifier.df.node.NumericalNode;

/**
 * This tool is to visualize the Decision tree
 */
public final class TreeVisualizer {
  
  private TreeVisualizer() {}
  
  private static String doubleToString(double value) {
    DecimalFormat df = new DecimalFormat("0.##");
    return df.format(value);
  }
  
  private static String toStringNode(Node node, Dataset dataset,
      String[] attrNames, Map<String,Field> fields, int layer) {
    
    StringBuilder buff = new StringBuilder();
    
    try {
      if (node instanceof CategoricalNode) {
        CategoricalNode cnode = (CategoricalNode) node;
        int attr = (Integer) fields.get("CategoricalNode.attr").get(cnode);
        double[] values = (double[]) fields.get("CategoricalNode.values").get(cnode);
        Node[] childs = (Node[]) fields.get("CategoricalNode.childs").get(cnode);
        String[][] attrValues = (String[][]) fields.get("Dataset.values").get(dataset);
        for (int i = 0; i < attrValues[attr].length; i++) {
          int index = ArrayUtils.indexOf(values, i);
          if (index < 0) {
            continue;
          }
          buff.append('\n');
          for (int j = 0; j < layer; j++) {
            buff.append("|   ");
          }
          buff.append(attrNames == null ? attr : attrNames[attr]).append(" = ")
              .append(attrValues[attr][i]);
          buff.append(toStringNode(childs[index], dataset, attrNames, fields, layer + 1));
        }
      } else if (node instanceof NumericalNode) {
        NumericalNode nnode = (NumericalNode) node;
        int attr = (Integer) fields.get("NumericalNode.attr").get(nnode);
        double split = (Double) fields.get("NumericalNode.split").get(nnode);
        Node loChild = (Node) fields.get("NumericalNode.loChild").get(nnode);
        Node hiChild = (Node) fields.get("NumericalNode.hiChild").get(nnode);
        buff.append('\n');
        for (int j = 0; j < layer; j++) {
          buff.append("|   ");
        }
        buff.append(attrNames == null ? attr : attrNames[attr]).append(" < ")
            .append(doubleToString(split));
        buff.append(toStringNode(loChild, dataset, attrNames, fields, layer + 1));
        buff.append('\n');
        for (int j = 0; j < layer; j++) {
          buff.append("|   ");
        }
        buff.append(attrNames == null ? attr : attrNames[attr]).append(" >= ")
            .append(doubleToString(split));
        buff.append(toStringNode(hiChild, dataset, attrNames, fields, layer + 1));
      } else if (node instanceof Leaf) {
        Leaf leaf = (Leaf) node;
        double label = (Double) fields.get("Leaf.label").get(leaf);
        if (dataset.isNumerical(dataset.getLabelId())) {
          buff.append(" : ").append(doubleToString(label));
        } else {
          buff.append(" : ").append(dataset.getLabelString(label));
        }
      }
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    }
    
    return buff.toString();
  }
  
  private static Map<String,Field> getReflectMap() {
    Map<String,Field> fields = new HashMap<String,Field>();
    
    try {
      Field m = CategoricalNode.class.getDeclaredField("attr");
      m.setAccessible(true);
      fields.put("CategoricalNode.attr", m);
      m = CategoricalNode.class.getDeclaredField("values");
      m.setAccessible(true);
      fields.put("CategoricalNode.values", m);
      m = CategoricalNode.class.getDeclaredField("childs");
      m.setAccessible(true);
      fields.put("CategoricalNode.childs", m);
      m = NumericalNode.class.getDeclaredField("attr");
      m.setAccessible(true);
      fields.put("NumericalNode.attr", m);
      m = NumericalNode.class.getDeclaredField("split");
      m.setAccessible(true);
      fields.put("NumericalNode.split", m);
      m = NumericalNode.class.getDeclaredField("loChild");
      m.setAccessible(true);
      fields.put("NumericalNode.loChild", m);
      m = NumericalNode.class.getDeclaredField("hiChild");
      m.setAccessible(true);
      fields.put("NumericalNode.hiChild", m);
      m = Leaf.class.getDeclaredField("label");
      m.setAccessible(true);
      fields.put("Leaf.label", m);
      m = Dataset.class.getDeclaredField("values");
      m.setAccessible(true);
      fields.put("Dataset.values", m);
    } catch (NoSuchFieldException nsfe) {
      throw new IllegalStateException(nsfe);
    }
    
    return fields;
  }
  
  /**
   * Decision tree to String
   * 
   * @param tree
   *          Node of tree
   * @param attrNames
   *          attribute names
   */
  public static String toString(Node tree, Dataset dataset, String[] attrNames) {
    return toStringNode(tree, dataset, attrNames, getReflectMap(), 0);
  }
  
  /**
   * Print Decision tree
   * 
   * @param tree
   *          Node of tree
   * @param attrNames
   *          attribute names
   */
  public static void print(Node tree, Dataset dataset, String[] attrNames) {
    System.out.println(toString(tree, dataset, attrNames));
  }
  
  private static String toStringPredict(Node node, Instance instance,
      Dataset dataset, String[] attrNames, Map<String,Field> fields) {
    StringBuilder buff = new StringBuilder();
    
    try {
      if (node instanceof CategoricalNode) {
        CategoricalNode cnode = (CategoricalNode) node;
        int attr = (Integer) fields.get("CategoricalNode.attr").get(cnode);
        double[] values = (double[]) fields.get("CategoricalNode.values").get(
            cnode);
        Node[] childs = (Node[]) fields.get("CategoricalNode.childs")
            .get(cnode);
        String[][] attrValues = (String[][]) fields.get("Dataset.values").get(
            dataset);
        
        int index = ArrayUtils.indexOf(values, instance.get(attr));
        if (index >= 0) {
          buff.append(attrNames == null ? attr : attrNames[attr]).append(" = ")
              .append(attrValues[attr][(int) instance.get(attr)]);
          buff.append(" -> ");
          buff.append(toStringPredict(childs[index], instance, dataset,
              attrNames, fields));
        }
      } else if (node instanceof NumericalNode) {
        NumericalNode nnode = (NumericalNode) node;
        int attr = (Integer) fields.get("NumericalNode.attr").get(nnode);
        double split = (Double) fields.get("NumericalNode.split").get(nnode);
        Node loChild = (Node) fields.get("NumericalNode.loChild").get(nnode);
        Node hiChild = (Node) fields.get("NumericalNode.hiChild").get(nnode);
        
        if (instance.get(attr) < split) {
          buff.append('(').append(attrNames == null ? attr : attrNames[attr])
              .append(" = ").append(doubleToString(instance.get(attr)))
              .append(") < ").append(doubleToString(split));
          buff.append(" -> ");
          buff.append(toStringPredict(loChild, instance, dataset, attrNames,
              fields));
        } else {
          buff.append('(').append(attrNames == null ? attr : attrNames[attr])
              .append(" = ").append(doubleToString(instance.get(attr)))
              .append(") >= ").append(doubleToString(split));
          buff.append(" -> ");
          buff.append(toStringPredict(hiChild, instance, dataset, attrNames,
              fields));
        }
      } else if (node instanceof Leaf) {
        Leaf leaf = (Leaf) node;
        double label = (Double) fields.get("Leaf.label").get(leaf);
        if (dataset.isNumerical(dataset.getLabelId())) {
          buff.append(doubleToString(label));
        } else {
          buff.append(dataset.getLabelString(label));
        }
      }
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    }
    
    return buff.toString();
  }
  
  /**
   * Predict trace to String
   * 
   * @param tree
   *          Node of tree
   * @param attrNames
   *          attribute names
   */
  public static String[] predictTrace(Node tree, Data data, String[] attrNames) {
    Map<String,Field> reflectMap = getReflectMap();
    String[] prediction = new String[data.size()];
    for (int i = 0; i < data.size(); i++) {
      prediction[i] = toStringPredict(tree, data.get(i), data.getDataset(),
          attrNames, reflectMap);
    }
    return prediction;
  }
  
  /**
   * Print predict trace
   * 
   * @param tree
   *          Node of tree
   * @param attrNames
   *          attribute names
   */
  public static void predictTracePrint(Node tree, Data data, String[] attrNames) {
    Map<String,Field> reflectMap = getReflectMap();
    for (int i = 0; i < data.size(); i++) {
      System.out.println(toStringPredict(tree, data.get(i), data.getDataset(),
          attrNames, reflectMap));
    }
  }
}
