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

package org.apache.mahout.utils.vectors.arff;

import com.google.common.collect.Maps;

import java.text.DateFormat;
import java.text.NumberFormat;
import java.text.ParseException;
import java.text.ParsePosition;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Holds ARFF information in {@link Map}.
 */
public class MapBackedARFFModel implements ARFFModel {
  
  private static final Pattern QUOTE_PATTERN = Pattern.compile("\"");
  
  private long wordCount = 1;
  
  private String relation;
  
  private final Map<String,Integer> labelBindings;
  private final Map<Integer,String> idxLabel;
  private final Map<Integer,ARFFType> typeMap; // key is the vector index, value is the type
  private final Map<Integer,DateFormat> dateMap;
  private final Map<String,Map<String,Integer>> nominalMap;
  private final Map<String,Long> words;
  
  public MapBackedARFFModel() {
    this(new HashMap<String,Long>(), 1, new HashMap<String,Map<String,Integer>>());
  }
  
  public MapBackedARFFModel(Map<String,Long> words, long wordCount, Map<String,Map<String,Integer>> nominalMap) {
    this.words = words;
    this.wordCount = wordCount;
    labelBindings = Maps.newHashMap();
    idxLabel = Maps.newHashMap();
    typeMap = Maps.newHashMap();
    dateMap = Maps.newHashMap();
    this.nominalMap = nominalMap;
    
  }
  
  @Override
  public String getRelation() {
    return relation;
  }
  
  @Override
  public void setRelation(String relation) {
    this.relation = relation;
  }
  
  /**
   * Convert a piece of String data at a specific spot into a value
   * 
   * @param data
   *          The data to convert
   * @param idx
   *          The position in the ARFF data
   * @return A double representing the data
   */
  @Override
  public double getValue(String data, int idx) {
    ARFFType type = typeMap.get(idx);
    if (type == null) {
      throw new IllegalArgumentException("Attribute type cannot be NULL, attribute index was: " + idx);
    }
    data = QUOTE_PATTERN.matcher(data).replaceAll("");
    data = data.trim();
    double result;
    switch (type) {
      case NUMERIC:
      case INTEGER:
      case REAL:
        result = processNumeric(data);
        break;
      case DATE:
        result = processDate(data, idx);
        break;
      case STRING:
        // may have quotes
        result = processString(data);
        break;
      case NOMINAL:
        String label = idxLabel.get(idx);
        result = processNominal(label, data);
        break;
      default:
        throw new IllegalStateException("Unknown type: " + type);
    }
    return result;
  }
  
  protected double processNominal(String label, String data) {
    double result;
    Map<String,Integer> classes = nominalMap.get(label);
    if (classes != null) {
      Integer ord = classes.get(ARFFType.removeQuotes(data));
      if (ord != null) {
        result = ord;
      } else {
        throw new IllegalStateException("Invalid nominal: " + data + " for label: " + label);
      }
    } else {
      throw new IllegalArgumentException("Invalid nominal label: " + label + " Data: " + data);
    }
    
    return result;
  }

  // Not sure how scalable this is going to be
  protected double processString(String data) {
    data = QUOTE_PATTERN.matcher(data).replaceAll("");
    // map it to an long
    Long theLong = words.get(data);
    if (theLong == null) {
      theLong = wordCount++;
      words.put(data, theLong);
    }
    return theLong;
  }
  
  protected static double processNumeric(String data) {
    if (isNumeric(data)) {
      return Double.parseDouble(data);
    }
    return Double.NaN;
  }

  public static boolean isNumeric(String str) {
    NumberFormat formatter = NumberFormat.getInstance();
    ParsePosition parsePosition = new ParsePosition(0);
    formatter.parse(str, parsePosition);
    return str.length() == parsePosition.getIndex();
  }

  protected double processDate(String data, int idx) {
    DateFormat format = dateMap.get(idx);
    if (format == null) {
      format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.ENGLISH);
    }
    double result;
    try {
      Date date = format.parse(data);
      result = date.getTime(); // hmmm, what kind of loss casting long to double?
    } catch (ParseException e) {
      throw new IllegalArgumentException(e);
    }
    return result;
  }
  
  /**
   * The vector attributes (labels in Mahout speak), unmodifiable
   * 
   * @return the map
   */
  @Override
  public Map<String,Integer> getLabelBindings() {
    return Collections.unmodifiableMap(labelBindings);
  }
  
  /**
   * The map of types encountered
   * 
   * @return the map
   */
  public Map<Integer,ARFFType> getTypeMap() {
    return Collections.unmodifiableMap(typeMap);
  }
  
  /**
   * Map of Date formatters used
   * 
   * @return the map
   */
  public Map<Integer,DateFormat> getDateMap() {
    return Collections.unmodifiableMap(dateMap);
  }
  
  /**
   * Map nominals to ids. Should only be modified by calling {@link ARFFModel#addNominal(String, String, int)}
   * 
   * @return the map
   */
  @Override
  public Map<String,Map<String,Integer>> getNominalMap() {
    return nominalMap;
  }
  
  /**
   * Immutable map of words to the long id used for those words
   * 
   * @return The map
   */
  @Override
  public Map<String,Long> getWords() {
    return words;
  }
  
  @Override
  public Integer getNominalValue(String label, String nominal) {
    return nominalMap.get(label).get(nominal);
  }
  
  @Override
  public void addNominal(String label, String nominal, int idx) {
    Map<String,Integer> noms = nominalMap.get(label);
    if (noms == null) {
      noms = Maps.newHashMap();
      nominalMap.put(label, noms);
    }
    noms.put(nominal, idx);
  }
  
  @Override
  public DateFormat getDateFormat(Integer idx) {
    return dateMap.get(idx);
  }
  
  @Override
  public void addDateFormat(Integer idx, DateFormat format) {
    dateMap.put(idx, format);
  }
  
  @Override
  public Integer getLabelIndex(String label) {
    return labelBindings.get(label);
  }
  
  @Override
  public void addLabel(String label, Integer idx) {
    labelBindings.put(label, idx);
    idxLabel.put(idx, label);
  }
  
  @Override
  public ARFFType getARFFType(Integer idx) {
    return typeMap.get(idx);
  }
  
  @Override
  public void addType(Integer idx, ARFFType type) {
    typeMap.put(idx, type);
  }
  
  /**
   * The count of the number of words seen
   * 
   * @return the count
   */
  @Override
  public long getWordCount() {
    return wordCount;
  }
  
  @Override
  public int getLabelSize() {
    return labelBindings.size();
  }
}
