package org.apache.mahout.utils.regex;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
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


/**
 *
 *
 **/
public class RegexUtils {
  public static final RegexTransformer IDENTITY_TRANSFORMER = new IdentityTransformer();
  public static final RegexFormatter IDENTITY_FORMATTER = new IdentityFormatter();

  public static String extract(String line, Pattern pattern, List<Integer> groupsToKeep,
                               String separator, RegexTransformer transformer) {
    StringBuilder bldr = new StringBuilder();
    extract(line, bldr, pattern, groupsToKeep, separator, transformer);
    return bldr.toString();
  }

  public static void extract(String line, StringBuilder outputBuffer,
                             Pattern pattern, List<Integer> groupsToKeep, String separator,
                             RegexTransformer transformer) {
    if (transformer == null) {
      transformer = IDENTITY_TRANSFORMER;
    }
    Matcher matcher = pattern.matcher(line);
    String match;
    if (groupsToKeep.isEmpty() == false) {
      while (matcher.find() == true) {
        for (Integer groupNum : groupsToKeep) {
          match = matcher.group(groupNum);
          if (match != null) {
            outputBuffer.append(transformer.transformMatch(match)).append(separator);
          }
        }
      }
    } else {
      while (matcher.find() == true) {
        match = matcher.group();
        if (match != null) {
          outputBuffer.append(transformer.transformMatch(match)).append(separator);
        }
      }
    }
    //trim off the last separator, which is always there
    if (outputBuffer.length() > 0) {
      outputBuffer.setLength(outputBuffer.length() - separator.length());
    }
  }
}
