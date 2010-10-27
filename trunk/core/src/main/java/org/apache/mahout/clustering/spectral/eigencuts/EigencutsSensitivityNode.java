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

package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * This class allows the storage of computed sensitivities in an
 * unordered fashion, instead having each sensitivity track its
 * own (i, j) coordinate. Thus these objects can be stored as elements
 * in any list or, in particular, Writable array.
 */
public class EigencutsSensitivityNode implements Writable {
  
  private int row;
  private int column;
  private double sensitivity;
  
  public EigencutsSensitivityNode(int i, int j, double s) {
    row = i;
    column = j;
    sensitivity = s;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.row = in.readInt();
    this.column = in.readInt();
    this.sensitivity = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(row);
    out.writeInt(column);
    out.writeDouble(sensitivity);
  }

  public int getRow() {
    return row;
  }

  public int getColumn() {
    return column;
  }

  public double getSensitivity() {
    return sensitivity;
  }
}
