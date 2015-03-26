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

package org.apache.mahout.classifier;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public class RegressionResultAnalyzerTest extends MahoutTestCase {

  private static final Pattern p1 = Pattern.compile("Correlation coefficient *: *(.*)\n");
  private static final Pattern p2 = Pattern.compile("Mean absolute error *: *(.*)\n");
  private static final Pattern p3 = Pattern.compile("Root mean squared error *: *(.*)\n");
  private static final Pattern p4 = Pattern.compile("Predictable Instances *: *(.*)\n");
  private static final Pattern p5 = Pattern.compile("Unpredictable Instances *: *(.*)\n");
  private static final Pattern p6 = Pattern.compile("Total Regressed Instances *: *(.*)\n");
  
  private static double[] parseAnalysis(CharSequence analysis) {
    double[] results = new double[3];
    Matcher m = p1.matcher(analysis);
    if (m.find()) {
      results[0] = Double.parseDouble(m.group(1));
    } else {
      return null;
    }
    m = p2.matcher(analysis);
    if (m.find()) {
      results[1] = Double.parseDouble(m.group(1));
    } else {
      return null;
    }
    m = p3.matcher(analysis);
    if (m.find()) {
      results[2] = Double.parseDouble(m.group(1));
    } else {
      return null;
    }
    return results;
  }

  private static int[] parseAnalysisCount(CharSequence analysis) {
    int[] results = new int[3];
    Matcher m = p4.matcher(analysis);
    if (m.find()) {
      results[0] = Integer.parseInt(m.group(1));
    }
    m = p5.matcher(analysis);
    if (m.find()) {
      results[1] = Integer.parseInt(m.group(1));
    }
    m = p6.matcher(analysis);
    if (m.find()) {
      results[2] = Integer.parseInt(m.group(1));
    }
    return results;
  }
  
  @Test
  public void testAnalyze() {
    double[][] results = new double[10][2];

    for (int i = 0; i < results.length; i++) {
      results[i][0] = i;
      results[i][1] = i + 1;
    }
    RegressionResultAnalyzer analyzer = new RegressionResultAnalyzer();
    analyzer.setInstances(results);
    String analysis = analyzer.toString();
    assertArrayEquals(new double[]{1.0, 1.0, 1.0}, parseAnalysis(analysis), 0);

    for (int i = 0; i < results.length; i++) {
      results[i][1] = Math.sqrt(i);
    }
    analyzer = new RegressionResultAnalyzer();
    analyzer.setInstances(results);
    analysis = analyzer.toString();
    assertArrayEquals(new double[]{0.9573, 2.5694, 3.2848}, parseAnalysis(analysis), 0);

    for (int i = 0; i < results.length; i++) {
      results[i][0] = results.length - i;
    }
    analyzer = new RegressionResultAnalyzer();
    analyzer.setInstances(results);
    analysis = analyzer.toString();
    assertArrayEquals(new double[]{-0.9573, 4.1351, 5.1573}, parseAnalysis(analysis), 0);
  }

  @Test
  public void testUnpredictable() {
    double[][] results = new double[10][2];

    for (int i = 0; i < results.length; i++) {
      results[i][0] = i;
      results[i][1] = Double.NaN;
    }
    RegressionResultAnalyzer analyzer = new RegressionResultAnalyzer();
    analyzer.setInstances(results);
    String analysis = analyzer.toString();
    assertNull(parseAnalysis(analysis));
    assertArrayEquals(new int[]{0, 10, 10}, parseAnalysisCount(analysis));

    for (int i = 0; i < results.length - 3; i++) {
      results[i][1] = Math.sqrt(i);
    }
    analyzer = new RegressionResultAnalyzer();
    analyzer.setInstances(results);
    analysis = analyzer.toString();
    assertArrayEquals(new double[]{0.9552, 1.4526, 1.9345}, parseAnalysis(analysis), 0);
    assertArrayEquals(new int[]{7, 3, 10}, parseAnalysisCount(analysis));
  }
}
