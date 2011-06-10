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

package org.apache.mahout.vectorizer.encoders;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import java.util.List;
import java.util.Map;

/**
* Assigns integer codes to strings as they appear.
*/
public class Dictionary {
  private final Map<String, Integer> dict = Maps.newLinkedHashMap();

  public int intern(String s) {
    if (!dict.containsKey(s)) {
      dict.put(s, dict.size());
    }
    return dict.get(s);
  }

  public List<String> values() {
    // order of keySet is guaranteed to be insertion order
    return Lists.newArrayList(dict.keySet());
  }

  public int size() {
    return dict.size();
  }

  public static Dictionary fromList(Iterable<String> values) {
    Dictionary dict = new Dictionary();
    for (String value : values) {
      dict.intern(value);
    }
    return dict;
  }
}
