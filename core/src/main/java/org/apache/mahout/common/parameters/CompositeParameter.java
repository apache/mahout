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

package org.apache.mahout.common.parameters;

import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.ClassUtils;

/** A placeholder for some sort of class with more parameters. */
public class CompositeParameter<T extends Parametered> extends AbstractParameter<T> {
  
  public CompositeParameter(Class<T> type,
                            String prefix,
                            String name,
                            Configuration jobConf,
                            T defaultValue,
                            String description) {
    super(type, prefix, name, jobConf, defaultValue, description);
  }
  
  @Override
  public void createParameters(String prefix, Configuration jobConf) {
    get().createParameters(prefix, jobConf);
  }
  
  @Override
  public Collection<Parameter<?>> getParameters() {
    return get().getParameters();
  }
  
  @Override
  public void configure(Configuration jobConf) {
    get().configure(jobConf);
  }
  
  @Override
  public void setStringValue(String className) {
    set((T) ClassUtils.instantiateAs(className, Object.class));
  }
  
  @Override
  public String getStringValue() {
    if (get() == null) {
      return null;
    }
    return get().getClass().getName();
  }
}
