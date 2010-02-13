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

import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.HEAD;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

/**
 * Writable for holding data generated from the collocation discovery jobs. Depending on the job configuration
 * gram may be one or more words. In some contexts this is used to hold a complete ngram, while in others it
 * holds a part of an existing ngram (subgram). Tracks the frequency of the gram and its position in the ngram
 * in which is was found.
 */
public class Gram implements WritableComparable<Gram> {
  
  public static enum Type {
    HEAD,
    TAIL,
    UNIGRAM
  };
  
  private String gram;
  private int frequency;
  private Type type;
  
  public Gram() {

  }
  
  public Gram(Gram other) {
    this.gram = other.gram;
    this.frequency = other.frequency;
    this.type = other.type;
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
   * @param type
   *          whether the gram is at the head of its text unit or tail or unigram
   */
  public Gram(String ngram, Type type) {
    this(ngram, 1, type);
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
   * @param type
   *          whether the gram is at the head of its text unit or tail or unigram
   */
  public Gram(String ngram, int frequency, Type type) {
    this.gram = ngram;
    this.frequency = frequency;
    this.type = type;
  }
  
  /**
   * @return the gram is at the head of its text unit or tail or unigram.
   */
  public Type getType() {
    return this.type;
  }
  
  /**
   * @param part
   *          whether the gram is at the head of its text unit or tail or unigram
   */
  public void setType(Type type) {
    this.type = type;
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
    int typeValue = in.readUnsignedByte();
    
    if (typeValue == 0) {
      type = Type.TAIL;
    } else if (typeValue == 1) {
      type = Type.HEAD;
    } else {
      type = Type.UNIGRAM;
    }
    
    Text data = new Text();
    data.readFields(in);
    gram = data.toString();
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(frequency);
    
    if (type == Type.TAIL) {
      out.writeByte(0);
    } else if (type == Type.HEAD) {
      out.writeByte(1);
    } else {
      out.writeByte(2);
    }
    
    Text data = new Text(gram);
    data.write(out);
  }
  
  @Override
  public int compareTo(Gram other) {
    int ret = getString().compareTo(other.getString());
    if (ret != 0) {
      return ret;
    }
    
    if (this.type == Type.UNIGRAM && other.type != Type.UNIGRAM) {
      return -1;
    }
    
    if (this.type != Type.UNIGRAM && other.type == Type.UNIGRAM) {
      return 1;
    }
    
    if (this.type == Type.HEAD && other.type != Type.HEAD) {
      return -1;
    }
    
    if (this.type != Type.HEAD && other.type == Type.HEAD) {
      return 1;
    }
    
    return 0;
  }
  
  /** Generates hashcode, does not include frequency in the hash calculation */
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + (gram == null ? 0 : gram.hashCode());
    result = prime * result + (type == null ? 0 : type.hashCode());
    return result;
  }
  
  /**
   * Determines equality, does not include frequency in the equality calculation
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    Gram other = (Gram) obj;
    if (gram == null) {
      if (other.gram != null) {
        return false;
      }
    } else if (!gram.equals(other.gram)) {
      return false;
    }
    if (type == null) {
      if (other.type != null) {
        return false;
      }
    } else if (!type.equals(other.type)) {
      return false;
    }
    return true;
  }
  
  @Override
  public String toString() {
    return "'" + gram + "'[" + (type == Type.UNIGRAM ? "u" : type == Type.HEAD ? "h" : "t") + "]:"
           + frequency;
  }
  
}
