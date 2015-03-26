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

package org.apache.mahout.clustering.spectral;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * Represents a vertex within the affinity graph for Eigencuts.
 */
public class VertexWritable implements Writable {
  
  /** the row */
  private int i;
  
  /** the column */
  private int j;
  
  /** the value at this vertex */
  private double value;
  
  /** an extra type delimeter, can probably be null */
  private String type;
  
  public VertexWritable() {
  }

  public VertexWritable(int i, int j, double v, String t) {
    this.i = i;
    this.j = j;
    this.value = v;
    this.type = t;
  }
  
  public int getRow() {
    return i;
  }
  
  public void setRow(int i) {
    this.i = i;
  }
  
  public int getCol() {
    return j;
  }
  
  public void setCol(int j) { 
    this.j = j;
  }
  
  public double getValue() {
    return value;
  }
  
  public void setValue(double v) {
    this.value = v;
  }
  
  public String getType() {
    return type;
  }
  
  public void setType(String t) {
    this.type = t;
  }
  
  @Override
  public void readFields(DataInput arg0) throws IOException {
    this.i = arg0.readInt();
    this.j = arg0.readInt();
    this.value = arg0.readDouble();
    this.type = arg0.readUTF();
  }

  @Override
  public void write(DataOutput arg0) throws IOException {
    arg0.writeInt(i);
    arg0.writeInt(j);
    arg0.writeDouble(value);
    arg0.writeUTF(type);
  }

}
