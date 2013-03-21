/*
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

package org.apache.mahout.cf.taste.hadoop.slopeone;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.math.Varint;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

final class FullRunningAverageAndStdDevWritable implements Writable {
  
  private FullRunningAverageAndStdDev average;
  
  public FullRunningAverageAndStdDevWritable(FullRunningAverageAndStdDev average) {
    this.average = average;
  }

  @Override
  public String toString() {
    return String.valueOf(average.getAverage())
        + '\t' + average.getCount() + '\t' + average.getMk() + '\t' + average.getSk();
  }
  
  @Override
  public void write(DataOutput dataOutput) throws IOException {
    Varint.writeUnsignedVarInt(average.getCount(), dataOutput);
    dataOutput.writeDouble(average.getAverage());
    dataOutput.writeDouble(average.getMk());
    dataOutput.writeDouble(average.getSk());
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    int count = Varint.readUnsignedVarInt(dataInput);
    double diff = dataInput.readDouble();
    double mk = dataInput.readDouble();
    double sk = dataInput.readDouble();
    average = new FullRunningAverageAndStdDev(count, diff, mk, sk);
  }

}
