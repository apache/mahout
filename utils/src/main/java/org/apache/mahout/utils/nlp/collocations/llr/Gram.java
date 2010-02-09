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

package org.apache.mahout.utils.nlp.collocations.llr;

import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Position.HEAD;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

/**
 * Writable for holding data generated from the collocation discovery jobs.
 * Depending on the job configuration gram may be one or more words. In some
 * contexts this is used to hold a complete ngram, while in others it holds a
 * part of an existing ngram (subgram). Tracks the frequency of the gram and its
 * position in the ngram in which is was found.
 */
public class Gram implements WritableComparable<Gram> {
  
  public static enum Position {
    HEAD,
    TAIL
  };
  
  private String gram;
  private int frequency;
  private Position position;
  
  public Gram() {

  }
  
  public Gram(Gram other) {
    this.gram = other.gram;
    this.frequency = other.frequency;
    this.position = other.position;
  }
  
  /**
   * Create an gram that is at the head of its text unit with a frequency of 1
   * 
   * @param gram
   *          the gram string
   */
  public Gram(String ngram) {
    this(ngram, 1, HEAD);
  }
  
  /**
   * Create an gram with a frequency of 1
   * 
   * @param gram
   *          the gram string
   * @param part
   *          whether the gram is at the head of its text unit.
   */
  public Gram(String ngram, Position position) {
    this(ngram, 1, position);
  }
  
  /**
   * Create an gram with a frequency of 1
   * 
   * @param gram
   *          the gram string
   * @param part
   *          whether the gram is at the head of its text unit.
   */
  public Gram(String ngram, int frequency) {
    this(ngram, frequency, HEAD);
  }
  
  /**
   * 
   * @param gram
   *          the gram string
   * @param frequency
   *          the gram frequency
   * @param part
   *          whether the gram is at the head of its text unit.
   */
  public Gram(String ngram, int frequency, Position position) {
    this.gram = ngram;
    this.frequency = frequency;
    this.position = position;
  }
  
  /**
   * @return position of gram in the text unit.
   */
  public Position getPosition() {
    return this.position;
  }
  
  /**
   * @param part
   *          position of the gram in the text unit.
   */
  public void setPosition(Position position) {
    this.position = position;
  }
  
  /**
   * @return gram term string
   */
  public String getString() {
    return gram;
  }
  
  /**
   * @param gram
   *          gram term string
   */
  public void setString(String str) {
    this.gram = str;
  }
  
  /**
   * @return gram frequency
   * @return
   */
  public int getFrequency() {
    return frequency;
  }
  
  /**
   * @param frequency
   *          gram's frequency
   */
  public void setFrequency(int frequency) {
    this.frequency = frequency;
  }
  
  public void incrementFrequency(int i) {
    this.frequency += i;
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    frequency = in.readInt();
    boolean head = in.readBoolean();
    
    if (head) position = Position.HEAD;
    else position = Position.TAIL;
    Text data = new Text();
    data.readFields(in);
    gram = data.toString();
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(frequency);
    
    if (position == Position.HEAD) out.writeBoolean(true);
    else out.writeBoolean(false);
    
    Text data = new Text(gram);
    data.write(out);
  }
  
  @Override
  public int compareTo(Gram other) {
    int ret = getString().compareTo(other.getString());
    if (ret != 0) {
      return ret;
    }
    
    if (this.position == Position.HEAD && other.position != Position.HEAD) {
      return -1;
    }
    
    if (this.position != Position.HEAD && other.position == Position.HEAD) {
      return 1;
    }
    
    return 0;
  }
  
  /** Generates hashcode, does not include frequency in the hash calculation */
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((gram == null) ? 0 : gram.hashCode());
    result = prime * result + ((position == null) ? 0 : position.hashCode());
    return result;
  }
  
  /**
   * Determines equality, does not include frequency in the equality calculation
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    Gram other = (Gram) obj;
    if (gram == null) {
      if (other.gram != null) return false;
    } else if (!gram.equals(other.gram)) return false;
    if (position == null) {
      if (other.position != null) return false;
    } else if (!position.equals(other.position)) return false;
    return true;
  }
  
  @Override
  public String toString() {
    return "'" + gram + "'[" + (position == Position.HEAD ? "h" : "t") + "]:"
           + frequency;
  }
  
}
