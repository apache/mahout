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

package org.apache.mahout.classifier.df.data;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.mahout.classifier.df.DFUtils;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Contains informations about the attributes.
 */
public class Dataset implements Writable {

  /**
   * Attributes type
   */
  public enum Attribute {
    IGNORED,
    NUMERICAL,
    CATEGORICAL,
    LABEL;

    public boolean isNumerical() {
      return this == NUMERICAL;
    }

    public boolean isCategorical() {
      return this == CATEGORICAL;
    }

    public boolean isLabel() {
      return this == LABEL;
    }

    public boolean isIgnored() {
      return this == IGNORED;
    }
  }

  private Attribute[] attributes;

  /**
   * list of ignored attributes
   */
  private int[] ignored;

  /**
   * distinct values (CATEGORIAL attributes only)
   */
  private String[][] values;

  /**
   * index of the label attribute in the loaded data (without ignored attributed)
   */
  private int labelId;

  /**
   * number of instances in the dataset
   */
  private int nbInstances;

  private Dataset() {
  }

  /**
   * Should only be called by a DataLoader
   *
   * @param attrs  attributes description
   * @param values distinct values for all CATEGORICAL attributes
   */
  Dataset(Attribute[] attrs, List<String>[] values, int nbInstances, boolean regression) {
    validateValues(attrs, values);

    int nbattrs = countAttributes(attrs);

    // the label values are set apart
    attributes = new Attribute[nbattrs];
    this.values = new String[nbattrs][];
    ignored = new int[attrs.length - nbattrs]; // nbignored = total - nbattrs

    labelId = -1;
    int ignoredId = 0;
    int ind = 0;
    for (int attr = 0; attr < attrs.length; attr++) {
      if (attrs[attr].isIgnored()) {
        ignored[ignoredId++] = attr;
        continue;
      }

      if (attrs[attr].isLabel()) {
        if (labelId != -1) {
          throw new IllegalStateException("Label found more than once");
        }
        labelId = ind;
        if (regression) {
          attrs[attr] = Attribute.NUMERICAL;
        } else {
          attrs[attr] = Attribute.CATEGORICAL;
        }
      }

      if (attrs[attr].isCategorical() || (!regression && attrs[attr].isLabel())) {
        this.values[ind] = new String[values[attr].size()];
        values[attr].toArray(this.values[ind]);
      }

      attributes[ind++] = attrs[attr];
    }

    if (labelId == -1) {
      throw new IllegalStateException("Label not found");
    }

    this.nbInstances = nbInstances;
  }

  public int nbValues(int attr) {
    return values[attr].length;
  }

  public String[] labels() {
    return Arrays.copyOf(values[labelId], nblabels());
  }

  public int nblabels() {
    return values[labelId].length;
  }

  public int getLabelId() {
    return labelId;
  }

  public double getLabel(Instance instance) {
    return instance.get(getLabelId());
  }

  public int nbInstances() {
    return nbInstances;
  }

  /**
   * Returns the code used to represent the label value in the data
   *
   * @param label label's value to code
   * @return label's code
   */
  public int labelCode(String label) {
    return ArrayUtils.indexOf(values[labelId], label);
  }

  /**
   * Returns the label value in the data
   * This method can be used when the criterion variable is the categorical attribute.
   *
   * @param code label's code
   * @return label's value
   */
  public String getLabelString(double code) {
    // handle the case (prediction == -1)
    if (code == -1) {
      return "unknown";
    }
    return values[labelId][(int) code];
  }

  /**
   * Converts a token to its corresponding int code for a given attribute
   *
   * @param attr attribute's index
   */
  public int valueOf(int attr, String token) {
    Preconditions.checkArgument(!isNumerical(attr), "Only for CATEGORICAL attributes");
    Preconditions.checkArgument(values != null, "Values not found");
    return ArrayUtils.indexOf(values[attr], token);
  }

  public int[] getIgnored() {
    return ignored;
  }


  /**
   * @return number of attributes that are not IGNORED
   */
  private static int countAttributes(Attribute[] attrs) {
    int nbattrs = 0;

    for (Attribute attr : attrs) {
      if (!attr.isIgnored()) {
        nbattrs++;
      }
    }

    return nbattrs;
  }

  private static void validateValues(Attribute[] attrs, List<String>[] values) {
    Preconditions.checkArgument(attrs.length == values.length, "attrs.length != values.length");
    for (int attr = 0; attr < attrs.length; attr++) {
      Preconditions.checkArgument(!attrs[attr].isCategorical() || values[attr] != null,
          "values not found for attribute " + attr);
    }
  }

  /**
   * @return number of attributes
   */
  public int nbAttributes() {
    return attributes.length;
  }

  /**
   * Is this a numerical attribute ?
   *
   * @param attr index of the attribute to check
   * @return true if the attribute is numerical
   */
  public boolean isNumerical(int attr) {
    return attributes[attr].isNumerical();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Dataset)) {
      return false;
    }

    Dataset dataset = (Dataset) obj;

    if (!Arrays.equals(attributes, dataset.attributes)) {
      return false;
    }

    for (int attr = 0; attr < nbAttributes(); attr++) {
      if (!Arrays.equals(values[attr], dataset.values[attr])) {
        return false;
      }
    }

    return labelId == dataset.labelId && nbInstances == dataset.nbInstances;
  }

  @Override
  public int hashCode() {
    int hashCode = labelId + 31 * nbInstances;
    for (Attribute attr : attributes) {
      hashCode = 31 * hashCode + attr.hashCode();
    }
    for (String[] valueRow : values) {
      if (valueRow == null) {
        continue;
      }
      for (String value : valueRow) {
        hashCode = 31 * hashCode + value.hashCode();
      }
    }
    return hashCode;
  }

  /**
   * Loads the dataset from a file
   *
   * @throws java.io.IOException
   */
  public static Dataset load(Configuration conf, Path path) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    FSDataInputStream input = fs.open(path);
    try {
      return read(input);
    } finally {
      Closeables.closeQuietly(input);
    }
  }

  public static Dataset read(DataInput in) throws IOException {
    Dataset dataset = new Dataset();

    dataset.readFields(in);
    return dataset;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int nbAttributes = in.readInt();
    attributes = new Attribute[nbAttributes];
    for (int attr = 0; attr < nbAttributes; attr++) {
      String name = WritableUtils.readString(in);
      attributes[attr] = Attribute.valueOf(name);
    }

    ignored = DFUtils.readIntArray(in);

    // only CATEGORICAL attributes have values
    values = new String[nbAttributes][];
    for (int attr = 0; attr < nbAttributes; attr++) {
      if (attributes[attr].isCategorical()) {
        values[attr] = WritableUtils.readStringArray(in);
      }
    }

    labelId = in.readInt();
    nbInstances = in.readInt();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(attributes.length); // nb attributes
    for (Attribute attr : attributes) {
      WritableUtils.writeString(out, attr.name());
    }

    DFUtils.writeArray(out, ignored);

    // only CATEGORICAL attributes have values
    for (String[] vals : values) {
      if (vals != null) {
        WritableUtils.writeStringArray(out, vals);
      }
    }

    out.writeInt(labelId);
    out.writeInt(nbInstances);
  }

}
