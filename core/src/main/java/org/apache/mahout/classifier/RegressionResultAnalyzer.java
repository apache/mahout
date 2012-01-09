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

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.StringUtils;

/**
 * ResultAnalyzer captures the classification statistics and displays in a tabular manner
 */
public class RegressionResultAnalyzer {

  private static class Result {
    private final double actual;
    private final double result;
    Result(double actual, double result) {
      this.actual = actual;
      this.result = result;
    }
    double getActual() {
      return actual;
    }
    double getResult() {
      return result;
    }
  }
  
  private List<Result> results;
  
  /**
   * 
   * @param actual
   *          The actual answer
   * @param result
   *          The regression result
   */
  public void addInstance(double actual, double result) {
    if (results == null) {
      results = new ArrayList<Result>();
    }
    results.add(new Result(actual, result));
  }

  /**
   * 
   * @param results
   *          The results table
   */
  public void setInstances(double[][] results) {
    for (double[] res : results) {
      addInstance(res[0], res[1]);
    }
  }

  @Override
  public String toString() {
    double sumActual = 0.0;
    double sumActualSquared = 0.0;
    double sumResult = 0.0;
    double sumResultSquared = 0.0;
    double sumAbsolute = 0.0;
    double sumAbsoluteSquared = 0.0;

    for (Result res : results) {
      double actual = res.getActual();
      double result = res.getResult();
      sumActual += actual;
      sumActualSquared += actual * actual;
      sumResult += result;
      sumResultSquared += result * result;
      double absolute = Math.abs(actual - result);
      sumAbsolute += absolute;
      sumAbsoluteSquared += absolute * absolute;
    }
    
    double varActual = sumActualSquared - sumActual * sumActual / results.size();
    double varResult = sumResultSquared - sumResult * sumResult / results.size();
    double varAbsolute = sumResultSquared - sumActual * sumResult /  results.size();

    double correlation;
    if (varActual * varResult <= 0) {
      correlation = 0.0;
    } else {
      correlation = varAbsolute / Math.sqrt(varActual * varResult);
    }

    StringBuilder returnString = new StringBuilder();
    
    returnString.append("=======================================================\n");
    returnString.append("Summary\n");
    returnString.append("-------------------------------------------------------\n");

    NumberFormat decimalFormatter = new DecimalFormat("0.####");
    
    returnString.append(StringUtils.rightPad("Correlation coefficient", 40)).append(": ").append(
      StringUtils.leftPad(decimalFormatter.format(correlation), 10)).append('\n');
    returnString.append(StringUtils.rightPad("Mean absolute error", 40)).append(": ").append(
      StringUtils.leftPad(decimalFormatter.format(sumAbsolute / results.size()), 10)).append('\n');
    returnString.append(StringUtils.rightPad("Root mean squared error", 40)).append(": ").append(
      StringUtils.leftPad(decimalFormatter.format(Math.sqrt(sumAbsoluteSquared / results.size())),
        10)).append('\n');
    returnString.append(StringUtils.rightPad("Total Regressed Instances", 40)).append(": ").append(
      StringUtils.leftPad(Integer.toString(results.size()), 10)).append('\n');
    returnString.append('\n');

    return returnString.toString();
  }
}
