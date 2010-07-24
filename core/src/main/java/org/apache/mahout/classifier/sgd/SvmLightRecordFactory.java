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

package org.apache.mahout.classifier.sgd;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.base.Splitter;
import com.google.common.collect.*;
import org.apache.mahout.math.Vector;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.*;

/**
 * Converts data in SVM Light format to vectors.
 *
 * Use of this class proceeds in a few steps.
 * <ul>
 * <li> Construct the class.  In contrast to csv data, SVM Light's input
 * format implicitly tells you what the target variable is.  Moreover, it makes
 * little sense on most problems encoded in SVM Light format to restrict
 * the input variables.
 * <li> Optionally, you tell the parser object about the possible values of the target
 * variable.  If you don't do this then you probably should set the number of distinct
 * values so that the target variable values will be taken from a restricted range.
 * <li> To encode data as a vector, you present a line of input to the factory and it
 * mutates a vector that you provide.  The factory also retains trace information so
 * that it can approximately reverse engineer vectors later.
 * <li> After converting data, you can ask for an explanation of the data in terms of
 * terms and weights.  In order to explain a vector accurately, the factory needs to
 * have seen the particular values of categorical fields (typically during encoding vectors)
 * and needs to have a reasonably small number of collisions in the vector encoding.
 * </ul>
 *
 * SVM Light input has one example per line and each line looks like this:
 *
 * <target> <feature1>:<weight1> <feature2>:<weight2> <feature3>:<weight3> ...
 *
 */
public class SvmLightRecordFactory implements RecordFactory {
  private static final String INTERCEPT_TERM = "Intercept Term";

  private Splitter onSpaces = Splitter.on(" ").trimResults().omitEmptyStrings();
  private Splitter onColon = Splitter.on(":").trimResults();

  private Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();


  private RecordValueEncoder encoder;

  private int maxTargetValue = Integer.MAX_VALUE;
  private Dictionary targetDictionary;

  private boolean includeBiasTerm;
  private RecordValueEncoder biasEncoder;

  /**
   * Construct a parser for SVM Light data that encodes the parsed data in vector form.
   */
  public SvmLightRecordFactory() {
    targetDictionary = new Dictionary();
    encoder = new StaticWordValueEncoder("data");
    biasEncoder = new ConstantValueEncoder("Intercept Term");
  }

  /**
   * Defines the values and thus the encoding of values of the target variables.  Note
   * that any values of the target variable not present in this list will be given the
   * value of the last member of the list.
   * @param values  The values the target variable can have.
   */
  public void defineTargetCategories(List<String> values) {
    if (values.size() > maxTargetValue) {
      throw new IllegalArgumentException("Must have less than or equal to " + maxTargetValue + " categories for target variable, but found " + values.size());
    }

    if (maxTargetValue == Integer.MAX_VALUE) {
      maxTargetValue = values.size();
    }

    for (String value : values) {
      targetDictionary.intern(value);
    }
  }

  /**
   * Defines the number of target variable categories, but allows this parser to
   * pick encodings for them as they appear.
   * @param max  The number of categories that will be excpeted.  Once this many have been
   * seen, all others will get the encoding max-1.
   */
  public SvmLightRecordFactory maxTargetValue(int max) {
    maxTargetValue = max;
    return this;
  }

  @Override
  public boolean usesFirstLineAsSchema() {
    return false;
  }

  /**
   * Decodes a single line of csv data and records the target and predictor variables in a record.
   * As a side effect, features are added into the featureVector.  Returns the value of the target
   * variable.
   *
   * @param line          The raw data.
   * @param featureVector Where to fill in the features.  Should be zeroed before calling
   *                      processLine.
   * @return The value of the target variable.
   */
  public int processLine(String line, Vector featureVector) {

    Iterator<String> values = onSpaces.split(line).iterator();

    String target = values.next();

    int targetValue = targetDictionary.intern(target);
    if (targetValue >= maxTargetValue) {
      targetValue = maxTargetValue - 1;
    }

    while (values.hasNext()) {
      Iterator<String> value = onColon.split(values.next()).iterator();
      String name = value.next();
      String weight = value.next();
      encoder.addToVector(name, Double.parseDouble(weight), featureVector);
    }
    if (includeBiasTerm) {
      biasEncoder.addToVector(null, featureVector);
    }
    return targetValue;
  }

  /**
   * Returns a list of the names of the predictor variables.
   *
   * @return A list of variable names.
   */
  public Iterable<String> getPredictors() {
    return ImmutableList.of("data", "Intercept Term");
  }

  public Map<String, Set<Integer>> getTraceDictionary() {
    return traceDictionary;
  }

  public SvmLightRecordFactory includeBiasTerm(boolean useBias) {
    includeBiasTerm = useBias;
    return this;
  }

  public List<String> getTargetCategories() {
    List<String> r = targetDictionary.values();
    if (r.size() > maxTargetValue) {
      r.subList(maxTargetValue, r.size()).clear();
    }
    return r;
  }

  @Override
  public void firstLine(String line) {
    throw new UnsupportedOperationException("SVM Light format doesn't have schema on first line");
  }

}
