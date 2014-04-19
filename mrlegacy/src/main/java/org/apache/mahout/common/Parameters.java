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

import java.io.IOException;
import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.util.GenericsUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Parameters {
  
  private static final Logger log = LoggerFactory.getLogger(Parameters.class);
  
  private Map<String,String> params = Maps.newHashMap();

  public Parameters() {

  }

  public Parameters(String serializedString) throws IOException {
    this(parseParams(serializedString));
  }

  protected Parameters(Map<String,String> params) {
    this.params = params;
  }

  public String get(String key) {
    return params.get(key);
  }
  
  public String get(String key, String defaultValue) {
    String ret = params.get(key);
    return ret == null ? defaultValue : ret;
  }
  
  public void set(String key, String value) {
    params.put(key, value);
  }

  public int getInt(String key, int defaultValue) {
    String ret = params.get(key);
    return ret == null ? defaultValue : Integer.parseInt(ret);
  }

  @Override
  public String toString() {
    Configuration conf = new Configuration();
    conf.set("io.serializations",
             "org.apache.hadoop.io.serializer.JavaSerialization,"
             + "org.apache.hadoop.io.serializer.WritableSerialization");
    DefaultStringifier<Map<String,String>> mapStringifier = new DefaultStringifier<Map<String,String>>(conf,
        GenericsUtil.getClass(params));
    try {
      return mapStringifier.toString(params);
    } catch (IOException e) {
      log.info("Encountered IOException while deserializing returning empty string", e);
      return "";
    }
    
  }
  
  public String print() {
    return params.toString();
  }

  public static Map<String,String> parseParams(String serializedString) throws IOException {
    Configuration conf = new Configuration();
    conf.set("io.serializations",
             "org.apache.hadoop.io.serializer.JavaSerialization,"
             + "org.apache.hadoop.io.serializer.WritableSerialization");
    Map<String,String> params = Maps.newHashMap();
    DefaultStringifier<Map<String,String>> mapStringifier = new DefaultStringifier<Map<String,String>>(conf,
        GenericsUtil.getClass(params));
    return mapStringifier.fromString(serializedString);
  }

}
