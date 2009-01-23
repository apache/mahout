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

package org.apache.mahout.ga.watchmaker.cd;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Fitness of the class discovery problem. 
 */
public class CDFitness implements Writable {

  /** True positive */
  private int tp;

  /** False positive */
  private int fp;

  /** True negative */
  private int tn;

  /** False negative */
  private int fn;

  public CDFitness() {

  }

  public CDFitness(CDFitness f) {
    tp = f.getTp();
    fp = f.getFp();
    tn = f.getTn();
    fn = f.getFn();
  }

  public CDFitness(int tp, int fp, int tn, int fn) {
    this.tp = tp;
    this.fp = fp;
    this.tn = tn;
    this.fn = fn;
  }

  public int getTp() {
    return tp;
  }

  public int getFp() {
    return fp;
  }

  public int getTn() {
    return tn;
  }

  public int getFn() {
    return fn;
  }

  public void add(CDFitness f) {
    tp += f.getTp();
    fp += f.getFp();
    tn += f.getTn();
    fn += f.getFn();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null || !(obj instanceof CDFitness))
      return false;

    CDFitness f = (CDFitness) obj;

    return tp == f.tp && fp == f.fp && tn == f.tn && fn == f.fn;
  }

  @Override
  public int hashCode() {
    return tp + 31 * (fp + 31 * (tn + 31 * fn));    
  }

  @Override
  public String toString() {
    return "[TP=" + tp + ", FP=" + fp + ", TN=" + tn + ", FN=" + fn + ']';
  }

  /**
   * Calculates the fitness corresponding to this evaluation.
   */
  public double get() {
    double se = ((double) tp) / (tp + fn); // sensitivity
    double sp = ((double) tn) / (tn + fp); // specificity
    
    return se * sp;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    tp = in.readInt();
    fp = in.readInt();
    tn = in.readInt();
    fn = in.readInt();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(tp);
    out.writeInt(fp);
    out.writeInt(tn);
    out.writeInt(fn);

  }

}
