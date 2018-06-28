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

package org.apache.mahout.vectorizer.collocations.llr;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;

import com.google.common.base.Preconditions;
import org.apache.hadoop.io.BinaryComparable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.Varint;

/**
 * Writable for holding data generated from the collocation discovery jobs. Depending on the job configuration
 * gram may be one or more words. In some contexts this is used to hold a complete ngram, while in others it
 * holds a part of an existing ngram (subgram). Tracks the frequency of the gram and its position in the ngram
 * in which is was found.
 */
public class Gram extends BinaryComparable implements WritableComparable<BinaryComparable> {
  
  public enum Type {
    HEAD('h'),
    TAIL('t'),
    UNIGRAM('u'),
    NGRAM('n');
    
    private final char x;
    
    Type(char c) {
      this.x = c;
    }

    @Override
    public String toString() {
      return String.valueOf(x);
    }
  }

  private byte[] bytes;
  private int length;
  private int frequency;
  
  public Gram() {

  }
  
  /**
   * Copy constructor
   */
  public Gram(Gram other) {
    frequency = other.frequency;
    length = other.length;
    bytes = other.bytes.clone();
  }

  /**
   * Create an gram with a frequency of 1
   * 
   * @param ngram
   *          the gram string
   * @param type
   *          whether the gram is at the head or tail of its text unit or it is a unigram
   */
  public Gram(String ngram, Type type) {
    this(ngram, 1, type);
  }
  

  /**
   * 
   * Create a gram with the specified frequency.
   * 
   * @param ngram
   *          the gram string
   * @param frequency
   *          the gram frequency
   * @param type
   *          whether the gram is at the head of its text unit or tail or unigram
   */
  public Gram(String ngram, int frequency, Type type) {
    Preconditions.checkNotNull(ngram);
    try {  
      // extra character is used for storing type which is part 
      // of the sort key.
      ByteBuffer bb = Text.encode('\0' + ngram, true);
      bytes = bb.array();
      length = bb.limit();
    } catch (CharacterCodingException e) {
      throw new IllegalStateException("Should not have happened ",e);
    }
    
    encodeType(type, bytes, 0);
    this.frequency = frequency;
  }
  
  
  @Override
  public byte[] getBytes() {
    return bytes;
  }

  @Override
  public int getLength() {
    return length;
  }

  /**
   * @return the gram is at the head of its text unit or tail or unigram.
   */
  public Type getType() {
    return decodeType(bytes, 0);
  }

  /**
   * @return gram term string
   */
  public String getString() {
    try {
      return Text.decode(bytes, 1, length - 1);
    } catch (CharacterCodingException e) {
      throw new IllegalStateException("Should not have happened " + e);
    }
  }

  /**
   * @return gram frequency
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
    int newLength = Varint.readUnsignedVarInt(in);
    setCapacity(newLength, false);
    in.readFully(bytes, 0, newLength);
    int newFrequency = Varint.readUnsignedVarInt(in);
    length = newLength;
    frequency = newFrequency;
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeUnsignedVarInt(length, out);
    out.write(bytes, 0, length);
    Varint.writeUnsignedVarInt(frequency, out);
  }

  /* Cribbed from o.a.hadoop.io.Text:
   * Sets the capacity of this object to <em>at least</em>
   * {@code len} bytes. If the current buffer is longer,
   * then the capacity and existing content of the buffer are
   * unchanged. If {@code len} is larger
   * than the current capacity, this object's capacity is
   * increased to match.
   * @param len the number of bytes we need
   * @param keepData should the old data be kept
   */
  private void setCapacity(int len, boolean keepData) {
    len++; // extra byte to hold type
    if (bytes == null || bytes.length < len) {
      byte[] newBytes = new byte[len];
      if (bytes != null && keepData) {
        System.arraycopy(bytes, 0, newBytes, 0, length);
      }
      bytes = newBytes;
    }
  }

  @Override
  public String toString() {
    return '\'' + getString() + "'[" + getType() + "]:" + frequency;
  }
  
  public static void encodeType(Type type, byte[] buf, int offset) {
    switch (type) {
      case HEAD:
        buf[offset] = 0x1;
        break;
      case TAIL:
        buf[offset] = 0x2; 
        break;
      case UNIGRAM:
        buf[offset] = 0x3;
        break;
      case NGRAM:
        buf[offset] = 0x4;
        break;
      default:
        throw new IllegalStateException("switch/case problem in encodeType");  
    }
  }
  
  public static Type decodeType(byte[] buf, int offset) {
    switch (buf[offset]) {
      case 0x1:
        return Type.HEAD;
      case 0x2:
        return Type.TAIL;
      case 0x3:
        return Type.UNIGRAM;
      case 0x4:
        return Type.NGRAM;
      default:
        throw new IllegalStateException("switch/case problem in decodeType");
    }
  }
}
