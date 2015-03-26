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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import org.apache.commons.csv.CSVUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.ContinuousValueEncoder;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;
import org.apache.mahout.vectorizer.encoders.TextValueEncoder;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Converts CSV data lines to vectors.
 *
 * Use of this class proceeds in a few steps.
 * <ul>
 * <li> At construction time, you tell the class about the target variable and provide
 * a dictionary of the types of the predictor values.  At this point,
 * the class yet cannot decode inputs because it doesn't know the fields that are in the
 * data records, nor their order.
 * <li> Optionally, you tell the parser object about the possible values of the target
 * variable.  If you don't do this then you probably should set the number of distinct
 * values so that the target variable values will be taken from a restricted range.
 * <li> Later, when you get a list of the fields, typically from the first line of a CSV
 * file, you tell the factory about these fields and it builds internal data structures
 * that allow it to decode inputs.  The most important internal state is the field numbers
 * for various fields.  After this point, you can use the factory for decoding data.
 * <li> To encode data as a vector, you present a line of input to the factory and it
 * mutates a vector that you provide.  The factory also retains trace information so
 * that it can approximately reverse engineer vectors later.
 * <li> After converting data, you can ask for an explanation of the data in terms of
 * terms and weights.  In order to explain a vector accurately, the factory needs to
 * have seen the particular values of categorical fields (typically during encoding vectors)
 * and needs to have a reasonably small number of collisions in the vector encoding.
 * </ul>
 */
public class CsvRecordFactory implements RecordFactory {
  private static final String INTERCEPT_TERM = "Intercept Term";

  private static final Map<String, Class<? extends FeatureVectorEncoder>> TYPE_DICTIONARY =
          ImmutableMap.<String, Class<? extends FeatureVectorEncoder>>builder()
                  .put("continuous", ContinuousValueEncoder.class)
                  .put("numeric", ContinuousValueEncoder.class)
                  .put("n", ContinuousValueEncoder.class)
                  .put("word", StaticWordValueEncoder.class)
                  .put("w", StaticWordValueEncoder.class)
                  .put("text", TextValueEncoder.class)
                  .put("t", TextValueEncoder.class)
                  .build();

  private final Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();

  private int target;
  private final Dictionary targetDictionary;
  
  //Which column is  used for identify a CSV file line 
  private String idName;
  private int id = -1;

  private List<Integer> predictors;
  private Map<Integer, FeatureVectorEncoder> predictorEncoders;
  private int maxTargetValue = Integer.MAX_VALUE;
  private final String targetName;
  private final Map<String, String> typeMap;
  private List<String> variableNames;
  private boolean includeBiasTerm;
  private static final String CANNOT_CONSTRUCT_CONVERTER =
      "Unable to construct type converter... shouldn't be possible";

  /**
   * Parse a single line of CSV-formatted text.
   *
   * Separated to make changing this functionality for the entire class easier
   * in the future.
   * @param line - CSV formatted text
   * @return List<String>
   */
  private List<String> parseCsvLine(String line) {
    try {
      return Arrays.asList(CSVUtils.parseLine(line));
	   }
	   catch (IOException e) {
      List<String> list = Lists.newArrayList();
      list.add(line);
      return list;
   	}
  }

  private List<String> parseCsvLine(CharSequence line) {
    return parseCsvLine(line.toString());
  }

  /**
   * Construct a parser for CSV lines that encodes the parsed data in vector form.
   * @param targetName            The name of the target variable.
   * @param typeMap               A map describing the types of the predictor variables.
   */
  public CsvRecordFactory(String targetName, Map<String, String> typeMap) {
    this.targetName = targetName;
    this.typeMap = typeMap;
    targetDictionary = new Dictionary();
  }

  public CsvRecordFactory(String targetName, String idName, Map<String, String> typeMap) {
    this(targetName, typeMap);
    this.idName = idName;
  }

  /**
   * Defines the values and thus the encoding of values of the target variables.  Note
   * that any values of the target variable not present in this list will be given the
   * value of the last member of the list.
   * @param values  The values the target variable can have.
   */
  @Override
  public void defineTargetCategories(List<String> values) {
    Preconditions.checkArgument(
        values.size() <= maxTargetValue,
        "Must have less than or equal to " + maxTargetValue + " categories for target variable, but found "
            + values.size());
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
   * @param max  The number of categories that will be expected.  Once this many have been
   * seen, all others will get the encoding max-1.
   */
  @Override
  public CsvRecordFactory maxTargetValue(int max) {
    maxTargetValue = max;
    return this;
  }

  @Override
  public boolean usesFirstLineAsSchema() {
    return true;
  }

  /**
   * Processes the first line of a file (which should contain the variable names). The target and
   * predictor column numbers are set from the names on this line.
   *
   * @param line       Header line for the file.
   */
  @Override
  public void firstLine(String line) {
    // read variable names, build map of name -> column
    final Map<String, Integer> vars = Maps.newHashMap();
    variableNames = parseCsvLine(line);
    int column = 0;
    for (String var : variableNames) {
      vars.put(var, column++);
    }

    // record target column and establish dictionary for decoding target
    target = vars.get(targetName);
    
    // record id column
    if (idName != null) {
      id = vars.get(idName);
    }

    // create list of predictor column numbers
    predictors = Lists.newArrayList(Collections2.transform(typeMap.keySet(), new Function<String, Integer>() {
      @Override
      public Integer apply(String from) {
        Integer r = vars.get(from);
        Preconditions.checkArgument(r != null, "Can't find variable %s, only know about %s", from, vars);
        return r;
      }
    }));

    if (includeBiasTerm) {
      predictors.add(-1);
    }
    Collections.sort(predictors);

    // and map from column number to type encoder for each column that is a predictor
    predictorEncoders = Maps.newHashMap();
    for (Integer predictor : predictors) {
      String name;
      Class<? extends FeatureVectorEncoder> c;
      if (predictor == -1) {
        name = INTERCEPT_TERM;
        c = ConstantValueEncoder.class;
      } else {
        name = variableNames.get(predictor);
        c = TYPE_DICTIONARY.get(typeMap.get(name));
      }
      try {
        Preconditions.checkArgument(c != null, "Invalid type of variable %s,  wanted one of %s",
          typeMap.get(name), TYPE_DICTIONARY.keySet());
        Constructor<? extends FeatureVectorEncoder> constructor = c.getConstructor(String.class);
        Preconditions.checkArgument(constructor != null, "Can't find correct constructor for %s", typeMap.get(name));
        FeatureVectorEncoder encoder = constructor.newInstance(name);
        predictorEncoders.put(predictor, encoder);
        encoder.setTraceDictionary(traceDictionary);
      } catch (InstantiationException e) {
        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
      } catch (InvocationTargetException e) {
        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
      } catch (NoSuchMethodException e) {
        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
      }
    }
  }


  /**
   * Decodes a single line of CSV data and records the target and predictor variables in a record.
   * As a side effect, features are added into the featureVector.  Returns the value of the target
   * variable.
   *
   * @param line          The raw data.
   * @param featureVector Where to fill in the features.  Should be zeroed before calling
   *                      processLine.
   * @return The value of the target variable.
   */
  @Override
  public int processLine(String line, Vector featureVector) {
    List<String> values = parseCsvLine(line);

    int targetValue = targetDictionary.intern(values.get(target));
    if (targetValue >= maxTargetValue) {
      targetValue = maxTargetValue - 1;
    }

    for (Integer predictor : predictors) {
      String value;
      if (predictor >= 0) {
        value = values.get(predictor);
      } else {
        value = null;
      }
      predictorEncoders.get(predictor).addToVector(value, featureVector);
    }
    return targetValue;
  }
  
  /***
   * Decodes a single line of CSV data and records the target(if retrunTarget is true)
   * and predictor variables in a record. As a side effect, features are added into the featureVector.
   * Returns the value of the target variable. When used during classify against production data without
   * target value, the method will be called with returnTarget = false. 
   * @param line The raw data.
   * @param featureVector Where to fill in the features.  Should be zeroed before calling
   *                      processLine.
   * @param returnTarget whether process and return target value, -1 will be returned if false.
   * @return The value of the target variable.
   */
  public int processLine(CharSequence line, Vector featureVector, boolean returnTarget) {
    List<String> values = parseCsvLine(line);
    int targetValue = -1;
    if (returnTarget) {
      targetValue = targetDictionary.intern(values.get(target));
      if (targetValue >= maxTargetValue) {
        targetValue = maxTargetValue - 1;
      }
    }

    for (Integer predictor : predictors) {
      String value = predictor >= 0 ? values.get(predictor) : null;
      predictorEncoders.get(predictor).addToVector(value, featureVector);
    }
    return targetValue;
  }
  
  /***
   * Extract the raw target string from a line read from a CSV file.
   * @param line the line of content read from CSV file
   * @return the raw target value in the corresponding column of CSV line 
   */
  public String getTargetString(CharSequence line) {
    List<String> values = parseCsvLine(line);
    return values.get(target);

  }

  /***
   * Extract the corresponding raw target label according to a code 
   * @param code the integer code encoded during training process
   * @return the raw target label
   */  
  public String getTargetLabel(int code) {
    for (String key : targetDictionary.values()) {
      if (targetDictionary.intern(key) == code) {
        return key;
      }
    }
    return null;
  }
  
  /***
   * Extract the id column value from the CSV record
   * @param line the line of content read from CSV file
   * @return the id value of the CSV record
   */
  public String getIdString(CharSequence line) {
    List<String> values = parseCsvLine(line);
    return values.get(id);
  }

  /**
   * Returns a list of the names of the predictor variables.
   *
   * @return A list of variable names.
   */
  @Override
  public Iterable<String> getPredictors() {
    return Lists.transform(predictors, new Function<Integer, String>() {
      @Override
      public String apply(Integer v) {
        if (v >= 0) {
          return variableNames.get(v);
        } else {
          return INTERCEPT_TERM;
        }
      }
    });
  }

  @Override
  public Map<String, Set<Integer>> getTraceDictionary() {
    return traceDictionary;
  }

  @Override
  public CsvRecordFactory includeBiasTerm(boolean useBias) {
    includeBiasTerm = useBias;
    return this;
  }

  @Override
  public List<String> getTargetCategories() {
    List<String> r = targetDictionary.values();
    if (r.size() > maxTargetValue) {
      r.subList(maxTargetValue, r.size()).clear();
    }
    return r;
  }

  public String getIdName() {
    return idName;
  }

  public void setIdName(String idName) {
    this.idName = idName;
  }

}
