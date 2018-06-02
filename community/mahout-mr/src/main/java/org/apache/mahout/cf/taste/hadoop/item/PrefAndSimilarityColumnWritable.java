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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class PrefAndSimilarityColumnWritable implements Writable {

  private float prefValue;
  private Vector similarityColumn;

  public PrefAndSimilarityColumnWritable() {
  }

  public PrefAndSimilarityColumnWritable(float prefValue, Vector similarityColumn) {
    set(prefValue, similarityColumn);
  }

  public void set(float prefValue, Vector similarityColumn) {
    this.prefValue = prefValue;
    this.similarityColumn = similarityColumn;
  }

  public float getPrefValue() {
    return prefValue;
  }

  public Vector getSimilarityColumn() {
    return similarityColumn;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    prefValue = in.readFloat();
    VectorWritable vw = new VectorWritable();
    vw.readFields(in);
    similarityColumn = vw.get();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeFloat(prefValue);
    VectorWritable vw = new VectorWritable(similarityColumn);
    vw.setWritesLaxPrecision(true);
    vw.write(out);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof PrefAndSimilarityColumnWritable) {
      PrefAndSimilarityColumnWritable other = (PrefAndSimilarityColumnWritable) obj;
      return prefValue == other.prefValue && similarityColumn.equals(other.similarityColumn);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashFloat(prefValue) + 31 * similarityColumn.hashCode();
  }


}
