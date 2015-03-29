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
 * This class is a Writable implementation of the mahout.common.Pair
 * generic class. Since the generic types would also themselves have to
 * implement Writable, it made more sense to create a more specialized
 * version of the class altogether.
 * 
 * In essence, this can be treated as a single Vector Element.
 */
public class IntDoublePairWritable implements Writable {
  
  private int key;
  private double value;
  
  public IntDoublePairWritable() {
  }
  
  public IntDoublePairWritable(int k, double v) {
    this.key = k;
    this.value = v;
  }
  
  public void setKey(int k) {
    this.key = k;
  }
  
  public void setValue(double v) {
    this.value = v;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.key = in.readInt();
    this.value = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(key);
    out.writeDouble(value);
  }

  public int getKey() {
    return key;
  }

  public double getValue() {
    return value;
  }

}
