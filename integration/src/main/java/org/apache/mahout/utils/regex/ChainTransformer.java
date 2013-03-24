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

package org.apache.mahout.utils.regex;

import com.google.common.collect.Lists;

import java.util.List;

/**
 * Chain together several {@link org.apache.mahout.utils.regex.RegexTransformer} and apply them to the match
 * in succession
 */
public class ChainTransformer implements RegexTransformer {

  private List<RegexTransformer> chain = Lists.newArrayList();

  public ChainTransformer() {
  }

  public ChainTransformer(List<RegexTransformer> chain) {
    this.chain = chain;
  }

  @Override
  public String transformMatch(String match) {
    String result = match;
    for (RegexTransformer transformer : chain) {
      result = transformer.transformMatch(result);
    }
    return result;
  }

  public List<RegexTransformer> getChain() {
    return chain;
  }

  public void setChain(List<RegexTransformer> chain) {
    this.chain = chain;
  }
}
