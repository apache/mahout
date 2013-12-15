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

import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public final class ARFFVectorIterableTest extends MahoutTestCase {

  @Test
  public void testValues() throws Exception {
    ARFFVectorIterable iterable = readModelFromResource("sample.arff");

    assertEquals("Mahout", iterable.getModel().getRelation());
    Map<String, Integer> bindings = iterable.getModel().getLabelBindings();
    assertNotNull(bindings);
    assertEquals(5, bindings.size());
    Iterator<Vector> iter = iterable.iterator();
    assertTrue(iter.hasNext());
    Vector next = iter.next();
    assertNotNull(next);
    assertTrue("Wrong instanceof", next instanceof DenseVector);
    assertEquals(1.0, next.get(0), EPSILON);
    assertEquals(2.0, next.get(1), EPSILON);
    assertTrue(iter.hasNext());
    next = iter.next();
    assertNotNull(next);
    assertTrue("Wrong instanceof", next instanceof DenseVector);
    assertEquals(2.0, next.get(0), EPSILON);
    assertEquals(3.0, next.get(1), EPSILON);

    assertTrue(iter.hasNext());
    next = iter.next();
    assertNotNull(next);
    assertTrue("Wrong instanceof", next instanceof RandomAccessSparseVector);
    assertEquals(5.0, next.get(0), EPSILON);
    assertEquals(23.0, next.get(1), EPSILON);

    assertFalse(iter.hasNext());
  }

  @Test
  public void testDense() throws Exception {
    Iterable<Vector> iterable = readModelFromResource("sample-dense.arff");
    Vector firstVector = iterable.iterator().next();
    assertEquals(1.0, firstVector.get(0), 0);
    assertEquals(65.0, firstVector.get(1), 0);
    assertEquals(1.0, firstVector.get(3), 0);
    assertEquals(1.0, firstVector.get(4), 0);

    int count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof DenseVector);
      count++;
    }
    assertEquals(5, count);
  }

  @Test
  public void testSparse() throws Exception {
    Iterable<Vector> iterable = readModelFromResource("sample-sparse.arff");

    Vector firstVector = iterable.iterator().next();
    assertEquals(23.1, firstVector.get(1), 0);
    assertEquals(3.23, firstVector.get(2), 0);
    assertEquals(1.2, firstVector.get(3), 0);

    int count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof RandomAccessSparseVector);
      count++;
    }
    assertEquals(9, count);
  }

  @Test
  public void testNonNumeric() throws Exception {
    MapBackedARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterable iterable = getVectors("non-numeric-1.arff", model);
    int count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof RandomAccessSparseVector);
      count++;
    }

    iterable = getVectors("non-numeric-1.arff", model);
    Iterator<Vector> iter = iterable.iterator();
    Vector firstVector = iter.next();

    assertEquals(1.0, firstVector.get(2), 0);

    assertEquals(10, count);
    Map<String, Map<String, Integer>> nominalMap = iterable.getModel().getNominalMap();
    assertNotNull(nominalMap);
    assertEquals(1, nominalMap.size());
    Map<String, Integer> noms = nominalMap.get("bar");
    assertNotNull("nominals for bar are null", noms);
    assertEquals(5, noms.size());
    Map<Integer, ARFFType> integerARFFTypeMap = model.getTypeMap();
    assertNotNull("Type map null", integerARFFTypeMap);
    assertEquals(5, integerARFFTypeMap.size());
    Map<String, Long> words = model.getWords();
    assertNotNull("words null", words);
    assertEquals(10, words.size());
    Map<Integer, DateFormat> integerDateFormatMap = model.getDateMap();
    assertNotNull("date format null", integerDateFormatMap);
    assertEquals(1, integerDateFormatMap.size());
  }

  @Test
  public void testDate() throws Exception {
    ARFFVectorIterable iterable = readModelFromResource("date.arff");
    Iterator<Vector> iter = iterable.iterator();
    Vector firstVector = iter.next();

    DateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.ENGLISH);
    Date date = format.parse("2001-07-04T12:08:56");
    long result = date.getTime();
    assertEquals(result, firstVector.get(1), 0);

    format = new SimpleDateFormat("yyyy.MM.dd G 'at' HH:mm:ss z", Locale.ENGLISH);
    date = format.parse("2001.07.04 AD at 12:08:56 PDT");
    result = date.getTime();
    assertEquals(result, firstVector.get(2), 0);

    format = new SimpleDateFormat("EEE, MMM d, ''yy", Locale.ENGLISH);
    date = format.parse("Wed, Jul 4, '01,4 0:08 PM, PDT");
    result = date.getTime();
    assertEquals(result, firstVector.get(3), 0);

    format = new SimpleDateFormat("K:mm a, z", Locale.ENGLISH);
    date = format.parse("0:08 PM, PDT");
    result = date.getTime();
    assertEquals(result, firstVector.get(4), 0);

    format = new SimpleDateFormat("yyyyy.MMMMM.dd GGG hh:mm aaa", Locale.ENGLISH);
    date = format.parse("02001.July.04 AD 12:08 PM");
    result = date.getTime();
    assertEquals(result, firstVector.get(5), 0);

    format = new SimpleDateFormat("EEE, d MMM yyyy HH:mm:ss Z", Locale.ENGLISH);
    date = format.parse("Wed, 4 Jul 2001 12:08:56 -0700");
    result = date.getTime();
    assertEquals(result, firstVector.get(6), 0);

  }

  @Test
  public void testMultipleNoms() throws Exception {
    MapBackedARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterable iterable = getVectors("non-numeric-1.arff", model);
    int count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof RandomAccessSparseVector);
      count++;
    }
    assertEquals(10, count);
    Map<String,Map<String,Integer>> nominalMap = iterable.getModel().getNominalMap();
    assertNotNull(nominalMap);
    assertEquals(1, nominalMap.size());
    Map<String,Integer> noms = nominalMap.get("bar");
    assertNotNull("nominals for bar are null", noms);
    assertEquals(5, noms.size());
    Map<Integer,ARFFType> integerARFFTypeMap = model.getTypeMap();
    assertNotNull("Type map null", integerARFFTypeMap);
    assertEquals(5, integerARFFTypeMap.size());
    Map<String,Long> words = model.getWords();
    assertNotNull("words null", words);
    assertEquals(10, words.size());

    Map<Integer,DateFormat> integerDateFormatMap = model.getDateMap();
    assertNotNull("date format null", integerDateFormatMap);
    assertEquals(1, integerDateFormatMap.size());


    iterable = getVectors("non-numeric-2.arff", model);
    count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof RandomAccessSparseVector);
      count++;
    }
    nominalMap = model.getNominalMap();
    assertNotNull(nominalMap);
    assertEquals(2, nominalMap.size());
    noms = nominalMap.get("test");
    assertNotNull("nominals for bar are null", noms);
    assertEquals(2, noms.size());
  }

  @Test
  public void testNumerics() throws Exception {
    String arff = "@RELATION numerics\n"
      + "@ATTRIBUTE theNumeric NUMERIC\n"
      + "@ATTRIBUTE theInteger INTEGER\n"
      + "@ATTRIBUTE theReal REAL\n"
      + "@DATA\n"
      + "1.0,2,3.0";
    ARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterable iterable = new ARFFVectorIterable(arff, model);
    model = iterable.getModel();
    assertNotNull(model);
    assertEquals(3, model.getLabelSize());
    assertEquals(ARFFType.NUMERIC, model.getARFFType(0));
    assertEquals(ARFFType.INTEGER, model.getARFFType(1));
    assertEquals(ARFFType.REAL, model.getARFFType(2));
    Iterator<Vector> it = iterable.iterator();
    Vector vector = it.next();
    assertEquals(1.0, vector.get(0), EPSILON);
    assertEquals(2.0, vector.get(1), EPSILON);
    assertEquals(3.0, vector.get(2), EPSILON);
  }

  @Test
  public void testQuotes() throws Exception {
    // ARFF allows quotes on identifiers
    ARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterable iterable = getVectors("quoted-id.arff", model);
    model = iterable.getModel();
    assertNotNull(model);
    assertEquals("quotes", model.getRelation());

    // check attribute labels
    assertEquals(4, model.getLabelSize());
    assertEquals(ARFFType.NUMERIC, model.getARFFType(0));
    assertEquals(ARFFType.INTEGER, model.getARFFType(1));
    assertEquals(ARFFType.REAL, model.getARFFType(2));
    assertEquals(ARFFType.NOMINAL, model.getARFFType(3));

    Map<String, Integer> labelBindings = model.getLabelBindings();
    assertTrue(labelBindings.keySet().contains("thenumeric"));
    assertTrue(labelBindings.keySet().contains("theinteger"));
    assertTrue(labelBindings.keySet().contains("thereal"));
    assertTrue(labelBindings.keySet().contains("thenominal"));

    // check nominal values
    Map<String, Integer> nominalMap = model.getNominalMap().get("thenominal");
    assertNotNull(nominalMap);
    assertEquals(3, nominalMap.size());
    assertTrue(nominalMap.keySet().contains("double-quote"));
    assertTrue(nominalMap.keySet().contains("single-quote"));
    assertTrue(nominalMap.keySet().contains("no-quote"));

    // check data values
    Iterator<Vector> it = iterable.iterator();
    Vector vector = it.next();
    assertEquals(nominalMap.get("no-quote"), vector.get(3), EPSILON);
    assertEquals(nominalMap.get("single-quote"), it.next().get(3), EPSILON);
    assertEquals(nominalMap.get("double-quote"), it.next().get(3), EPSILON);
  }

  static ARFFVectorIterable getVectors(String resourceName, ARFFModel model) throws IOException {
    String sample = Resources.toString(Resources.getResource(resourceName), Charsets.UTF_8);
    return new ARFFVectorIterable(sample, model);
  }

  private static ARFFVectorIterable readModelFromResource(String resourceName) throws IOException {
    ARFFModel model = new MapBackedARFFModel();
    return getVectors(resourceName, model);
  }

}
