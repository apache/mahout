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

package org.apache.mahout.fpm.pfpgrowth.convertors.string;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;

/**
 * A class which collects Top K string patterns
 *
 */
public final class TopKStringPatterns implements Writable {
  private final List<Pair<List<String>,Long>> frequentPatterns;
  
  public TopKStringPatterns() {
    frequentPatterns = Lists.newArrayList();
  }
  
  public TopKStringPatterns(Collection<Pair<List<String>, Long>> patterns) {
    frequentPatterns = Lists.newArrayList();
    frequentPatterns.addAll(patterns);
  }
  
  public Iterator<Pair<List<String>,Long>> iterator() {
    return frequentPatterns.iterator();
  }
  
  public List<Pair<List<String>,Long>> getPatterns() {
    return frequentPatterns;
  }
  
  public TopKStringPatterns merge(TopKStringPatterns pattern, int heapSize) {
    List<Pair<List<String>,Long>> patterns = Lists.newArrayList();
    Iterator<Pair<List<String>,Long>> myIterator = frequentPatterns.iterator();
    Iterator<Pair<List<String>,Long>> otherIterator = pattern.iterator();
    Pair<List<String>,Long> myItem = null;
    Pair<List<String>,Long> otherItem = null;
    for (int i = 0; i < heapSize; i++) {
      if (myItem == null && myIterator.hasNext()) {
        myItem = myIterator.next();
      }
      if (otherItem == null && otherIterator.hasNext()) {
        otherItem = otherIterator.next();
      }
      if (myItem != null && otherItem != null) {
        int cmp = myItem.getSecond().compareTo(otherItem.getSecond());
        if (cmp == 0) {
          cmp = myItem.getFirst().size() - otherItem.getFirst().size();
          if (cmp == 0) {
            for (int j = 0; j < myItem.getFirst().size(); j++) {
              cmp = myItem.getFirst().get(j).compareTo(
                otherItem.getFirst().get(j));
              if (cmp != 0) {
                break;
              }
            }
          }
        }
        if (cmp <= 0) {
          patterns.add(otherItem);
          if (cmp == 0) {
            myItem = null;
          }
          otherItem = null;
        } else if (cmp > 0) {
          patterns.add(myItem);
          myItem = null;
        }
      } else if (myItem != null) {
        patterns.add(myItem);
        myItem = null;
      } else if (otherItem != null) {
        patterns.add(otherItem);
        otherItem = null;
      } else {
        break;
      }
    }
    return new TopKStringPatterns(patterns);
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    frequentPatterns.clear();
    int length = in.readInt();
    for (int i = 0; i < length; i++) {
      List<String> items = Lists.newArrayList();
      int itemsetLength = in.readInt();
      long support = in.readLong();
      for (int j = 0; j < itemsetLength; j++) {
        items.add(in.readUTF());
      }
      frequentPatterns.add(new Pair<List<String>,Long>(items, support));
    }
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(frequentPatterns.size());
    for (Pair<List<String>,Long> pattern : frequentPatterns) {
      out.writeInt(pattern.getFirst().size());
      out.writeLong(pattern.getSecond());
      for (String item : pattern.getFirst()) {
        out.writeUTF(item);
      }
    }
  }
  
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    String sep = "";
    for (Pair<List<String>,Long> pattern : frequentPatterns) {
      sb.append(sep);
      sb.append(pattern.toString());
      sep = ", ";
      
    }
    return sb.toString();
    
  }
}
