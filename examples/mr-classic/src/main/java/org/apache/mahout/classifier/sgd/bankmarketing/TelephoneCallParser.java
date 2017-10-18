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

package org.apache.mahout.classifier.sgd.bankmarketing;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.collect.AbstractIterator;
import com.google.common.io.Resources;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;

/** Parses semi-colon separated data as TelephoneCalls  */
public class TelephoneCallParser implements Iterable<TelephoneCall> {

  private final Splitter onSemi = Splitter.on(";").trimResults(CharMatcher.anyOf("\" ;"));
  private String resourceName;

  public TelephoneCallParser(String resourceName) throws IOException {
    this.resourceName = resourceName;
  }

  @Override
  public Iterator<TelephoneCall> iterator() {
    try {
      return new AbstractIterator<TelephoneCall>() {
        BufferedReader input =
            new BufferedReader(new InputStreamReader(Resources.getResource(resourceName).openStream()));
        Iterable<String> fieldNames = onSemi.split(input.readLine());

          @Override
          protected TelephoneCall computeNext() {
            try {
              String line = input.readLine();
              if (line == null) {
                return endOfData();
              }

              return new TelephoneCall(fieldNames, onSemi.split(line));
            } catch (IOException e) {
              throw new RuntimeException("Error reading data", e);
            }
          }
        };
      } catch (IOException e) {
        throw new RuntimeException("Error reading data", e);
      }
  }
}
