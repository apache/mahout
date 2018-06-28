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
import java.nio.charset.CharacterCodingException;

import org.apache.hadoop.io.BinaryComparable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.Varint;
import org.apache.mahout.vectorizer.collocations.llr.Gram.Type;

/** A GramKey, based on the identity fields of Gram (type, string) plus a byte[] used for secondary ordering */
public final class GramKey extends BinaryComparable implements WritableComparable<BinaryComparable> {

  private int primaryLength;
  private int length;
  private byte[] bytes;
  
  public GramKey() {
    
  }
  
  /** create a GramKey based on the specified Gram and order
   * 
   * @param gram
   * @param order
   */
  public GramKey(Gram gram, byte[] order) {
    set(gram, order);
  }
  
  /** set the gram held by this key */
  public void set(Gram gram, byte[] order) {
    primaryLength = gram.getLength();
    length = primaryLength + order.length;
    setCapacity(length, false);
    System.arraycopy(gram.getBytes(), 0, bytes, 0, primaryLength);
    if (order.length > 0) {
      System.arraycopy(order, 0, bytes, primaryLength, order.length);
    }
  }

  @Override
  public byte[] getBytes() {
    return bytes;
  }

  @Override
  public int getLength() {
    return length;
  }

  public int getPrimaryLength() {
    return primaryLength;
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    int newLength = Varint.readUnsignedVarInt(in);
    int newPrimaryLength = Varint.readUnsignedVarInt(in);
    setCapacity(newLength, false);
    in.readFully(bytes, 0, newLength);
    length = newLength;
    primaryLength = newPrimaryLength;

  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeUnsignedVarInt(length, out);
    Varint.writeUnsignedVarInt(primaryLength, out);
    out.write(bytes, 0, length);
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
    if (bytes == null || bytes.length < len) {
      byte[] newBytes = new byte[len];
      if (bytes != null && keepData) {
        System.arraycopy(bytes, 0, newBytes, 0, length);
      }
      bytes = newBytes;
    }
  }
  
  /**
   * @return the gram is at the head of its text unit or tail or unigram.
   */
  public Type getType() {
    return Gram.decodeType(bytes, 0);
  }

  public String getPrimaryString() {
    try {
      return Text.decode(bytes, 1, primaryLength - 1);
    } catch (CharacterCodingException e) {
      throw new IllegalStateException(e);
    }
  }
  
  @Override
  public String toString() {
    return '\'' + getPrimaryString() + "'[" + getType() + ']';
  }
}
