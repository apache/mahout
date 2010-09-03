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

package org.apache.mahout.common;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang.mutable.MutableLong;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.Counters.Counter;

public final class DummyReporter implements Reporter {

  //private String status = "";
  private final Map<Enum<?>,MutableLong> count1 = new HashMap<Enum<?>,MutableLong>();
  private final Map<String,Map<String,MutableLong>> count2 = new HashMap<String,Map<String,MutableLong>>();
  
  @Override
  public Counter getCounter(Enum<?> name) {
    throw new NotImplementedException();
  }
  
  @Override
  public Counter getCounter(String group, String name) {
    throw new NotImplementedException();
  }
  
  @Override
  public InputSplit getInputSplit() throws UnsupportedOperationException {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public void incrCounter(Enum<?> key, long amount) {
    if (!count1.containsKey(key)) {
      count1.put(key, new MutableLong(0));
    }
    count1.get(key).add(amount);
  }
  
  @Override
  public void incrCounter(String group, String counter, long amount) {
    if (!count2.containsKey(group)) {
      count2.put(group, new HashMap<String,MutableLong>());
    }
    if (!count2.get(group).containsKey(counter)) {
      count2.get(group).put(counter, new MutableLong(0));
    }
    count2.get(group).get(counter).add(amount);
    
  }
  
  @Override
  public void setStatus(String status) {
    //this.status = status;
  }
  
  @Override
  public void progress() {

  }
  
}
