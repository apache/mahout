package org.apache.mahout.utils.vectors.arff;
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

import java.text.DateFormat;
import java.text.ParseException;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;


/**
 * Holds ARFF information in {@link java.util.Map}.
 */
public class MapBackedARFFModel implements ARFFModel {

  protected long wordCount = 1;

  protected String relation;

  private Map<String, Integer> labelBindings;
  private Map<Integer, String> idxLabel;
  private Map<Integer, ARFFType> typeMap; //key is the vector index, value is the type
  private Map<Integer, DateFormat> dateMap;
  private Map<String, Map<String, Integer>> nominalMap;
  private Map<String, Long> words;

  public MapBackedARFFModel() {
    this(new HashMap<String, Long>(), 1, new HashMap<String, Map<String, Integer>>());
  }

  public MapBackedARFFModel(Map<String, Long> words, long wordCount, Map<String, Map<String, Integer>> nominalMap) {
    this.words = words;
    this.wordCount = wordCount;
    labelBindings = new HashMap<String, Integer>();
    idxLabel = new HashMap<Integer, String>();
    typeMap = new HashMap<Integer, ARFFType>();
    dateMap = new HashMap<Integer, DateFormat>();
    this.nominalMap = nominalMap;

  }

  public String getRelation() {
    return relation;
  }

  public void setRelation(String relation) {
    this.relation = relation;
  }

  /**
   * Convert a piece of String data at a specific spot into a value
   *
   * @param data The data to convert
   * @param idx  The position in the ARFF data
   * @return A double representing the data
   */
  public double getValue(String data, int idx) {
    double result = 0;
    ARFFType type = typeMap.get(idx);
    data = data.replaceAll("\"", "");
    data = data.trim();
    switch (type) {
      case NUMERIC: {
        result = processNumeric(data);
        break;
      }
      case DATE: {
        result = processDate(data, idx);
        break;
      }
      case STRING: {
        //may have quotes
        result = processString(data);
        break;
      }
      case NOMINAL: {
        String label = idxLabel.get(idx);
        result = processNominal(label, data);
        break;
      }


    }
    return result;
  }

  protected double processNominal(String label, String data) {
    double result;
    Map<String, Integer> classes = nominalMap.get(label);
    if (classes != null) {
      Integer ord = classes.get(data);
      if (ord != null) {
        result = ord;
      } else {
        throw new RuntimeException("Invalid nominal: " + data + " for label: " + label);
      }
    } else {
      throw new RuntimeException("Invalid nominal label: " + label + " Data: " + data);
    }

    return result;
  }

  /**
   * Process a String
   *
   * @param data
   * @return
   */
  //Not sure how scalable this is going to be
  protected double processString(String data) {
    double result;
    data = data.replaceAll("\"", "");
    //map it to an long
    Long theLong = words.get(data);
    if (theLong == null) {
      theLong = wordCount++;
      words.put(data, theLong);
    }
    result = theLong;
    return result;
  }

  protected double processNumeric(String data) {
    return Double.parseDouble(data);
  }

  protected double processDate(String data, int idx) {
    double result;
    DateFormat format = dateMap.get(idx);
    if (format == null) {
      format = DEFAULT_DATE_FORMAT;
    }
    Date date = null;
    try {
      date = format.parse(data);
      result = date.getTime();// hmmm, what kind of loss casting long to double?
    } catch (ParseException e) {
      throw new RuntimeException(e);
    }
    return result;
  }

  /**
   * The vector attributes (labels in Mahout speak), unmodifiable
   * @return the map
   */
  public Map<String, Integer> getLabelBindings() {
    return Collections.unmodifiableMap(labelBindings);
  }

  /**
   * The map of types encountered
   * @return the map
   */
  public Map<Integer, ARFFType> getTypeMap() {
    return Collections.unmodifiableMap(typeMap);
  }

  /**
   * Map of Date formatters used
   * @return the map
   */
  public Map<Integer, DateFormat> getDateMap() {
    return Collections.unmodifiableMap(dateMap);
  }

  /**
   * Map nominals to ids.  Should only be modified by calling {@link ARFFModel#addNominal(String, String, int)}
   * @return the map
   */
  public Map<String, Map<String, Integer>> getNominalMap() {
    return nominalMap;
  }

  /**
   * Immutable map of words to the long id used for those words
   * @return The map
   */
  public Map<String, Long> getWords() {
    return words;
  }

  public Integer getNominalValue(String label, String nominal){
    return nominalMap.get(label).get(nominal);
  }

  public void addNominal(String label, String nominal, int idx) {
    Map<String, Integer> noms = nominalMap.get(label);
    if (noms == null) {
      noms = new HashMap<String, Integer>();
      nominalMap.put(label, noms);
    }
    noms.put(nominal, idx);
  }

  public DateFormat getDateFormat(Integer idx){
    return dateMap.get(idx);
  }

  public void addDateFormat(Integer idx, DateFormat format) {
    dateMap.put(idx, format);
  }

  public Integer getLabelIndex(String label){
    return labelBindings.get(label);
  }

  public void addLabel(String label, Integer idx) {
    labelBindings.put(label, idx);
    idxLabel.put(idx, label);
  }

  public ARFFType getARFFType(Integer idx){
    return typeMap.get(idx);
  }

  public void addType(Integer idx, ARFFType type) {
    typeMap.put(idx, type);
  }

  /**
   * The count of the number of words seen
   * @return the count
   */
  public long getWordCount() {
    return wordCount;
  }

  public int getLabelSize() {
    return labelBindings.size();
  }
}
