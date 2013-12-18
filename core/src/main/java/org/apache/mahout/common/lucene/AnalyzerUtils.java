package org.apache.mahout.common.lucene;
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

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.ClassUtils;

public final class AnalyzerUtils {

  private AnalyzerUtils() {}

  /**
   * Create an Analyzer using the latest {@link org.apache.lucene.util.Version}.  Note, if you need to pass in
   * parameters to your constructor, you will need to wrap it in an implementation that does not take any arguments
   * @param analyzerClassName - Lucene Analyzer Name
   * @return {@link Analyzer}
   * @throws ClassNotFoundException - {@link ClassNotFoundException}
   */
  public static Analyzer createAnalyzer(String analyzerClassName) throws ClassNotFoundException {
    return createAnalyzer(analyzerClassName, Version.LUCENE_46);
  }

  public static Analyzer createAnalyzer(String analyzerClassName, Version version) throws ClassNotFoundException {
    Class<? extends Analyzer> analyzerClass = Class.forName(analyzerClassName).asSubclass(Analyzer.class);
    return createAnalyzer(analyzerClass, version);
  }

  /**
   * Create an Analyzer using the latest {@link org.apache.lucene.util.Version}.  Note, if you need to pass in
   * parameters to your constructor, you will need to wrap it in an implementation that does not take any arguments
   * @param analyzerClass The Analyzer Class to instantiate
   * @return {@link Analyzer}
   */
  public static Analyzer createAnalyzer(Class<? extends Analyzer> analyzerClass) {
    return createAnalyzer(analyzerClass, Version.LUCENE_46);
  }

  public static Analyzer createAnalyzer(Class<? extends Analyzer> analyzerClass, Version version) {
    try {
      return ClassUtils.instantiateAs(analyzerClass, Analyzer.class,
          new Class<?>[] { Version.class }, new Object[] { version });
    } catch (IllegalStateException e) {
      return ClassUtils.instantiateAs(analyzerClass, Analyzer.class);
    }
  }
}
