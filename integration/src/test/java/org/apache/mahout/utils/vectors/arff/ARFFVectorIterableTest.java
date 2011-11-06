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

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.MahoutTestCase;
import org.junit.Test;

public final class ARFFVectorIterableTest extends MahoutTestCase {

  @Test
  public void testValues() throws Exception {
    StringBuilder builder = new StringBuilder();
    builder.append("%comments").append('\n').append("@RELATION Mahout").append('\n').append(
      "@ATTRIBUTE foo numeric").append('\n').append("@ATTRIBUTE bar numeric").append('\n').append(
      "@ATTRIBUTE timestamp DATE \"yyyy-MM-dd HH:mm:ss\"").append('\n').append("@ATTRIBUTE junk string")
        .append('\n').append("@ATTRIBUTE theNominal {c,b,a}").append('\n').append("@DATA").append('\n')
        .append("1,2, \"2009-01-01 5:55:55\", foo, c").append('\n').append("2,3").append('\n').append(
          "{0 5,1 23}").append('\n');
    ARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterable iterable = new ARFFVectorIterable(builder.toString(), model);
    assertEquals("Mahout", iterable.getModel().getRelation());
    Map<String,Integer> bindings = iterable.getModel().getLabelBindings();
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
    ARFFModel model = new MapBackedARFFModel();
    Iterable<Vector> iterable = new ARFFVectorIterable(SAMPLE_DENSE_ARFF, model);
    int count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof DenseVector);
      count++;
    }
    assertEquals(10, count);
  }

  @Test
  public void testSparse() throws Exception {
    ARFFModel model = new MapBackedARFFModel();
    Iterable<Vector> iterable = new ARFFVectorIterable(SAMPLE_SPARSE_ARFF, model);
    int count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof RandomAccessSparseVector);
      count++;
    }
    assertEquals(10, count);
  }

  @Test
  public void testNonNumeric() throws Exception {
    
    MapBackedARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterable iterable = new ARFFVectorIterable(NON_NUMERIC_ARFF, model);
    int count = 0;
    for (Vector vector : iterable) {
      assertTrue("Vector is not dense", vector instanceof RandomAccessSparseVector);
      count++;
    }
    
    iterable = new ARFFVectorIterable(NON_NUMERIC_ARFF, model);
    Iterator<Vector> iter = iterable.iterator();
    Vector firstVector = iter.next();
    
    assertEquals(1.0, firstVector.get(2), 0);
    
    assertEquals(10, count);
    Map<String,Map<String,Integer>> nominalMap = iterable.getModel().getNominalMap();
    assertNotNull(nominalMap);
    assertEquals(1, nominalMap.size());
    Map<String,Integer> noms = nominalMap.get("bar");
    assertNotNull("nominals for bar are null", noms);
    assertEquals(2, noms.size());
    Map<Integer,ARFFType> integerARFFTypeMap = model.getTypeMap();
    assertNotNull("Type map null", integerARFFTypeMap);
    assertEquals(5, integerARFFTypeMap.size());
    Map<String,Long> words = model.getWords();
    assertNotNull("words null", words);
    assertEquals(10, words.size());
    // System.out.println("Words: " + words);
    Map<Integer,DateFormat> integerDateFormatMap = model.getDateMap();
    assertNotNull("date format null", integerDateFormatMap);
    assertEquals(1, integerDateFormatMap.size());
  }

  @Test
  public void testDate() throws Exception {
    MapBackedARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterable iterable = new ARFFVectorIterable(DATE_ARFF, model);
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
    ARFFVectorIterable iterable = new ARFFVectorIterable(NON_NUMERIC_ARFF, model);
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
    assertEquals(2, noms.size());
    Map<Integer,ARFFType> integerARFFTypeMap = model.getTypeMap();
    assertNotNull("Type map null", integerARFFTypeMap);
    assertEquals(5, integerARFFTypeMap.size());
    Map<String,Long> words = model.getWords();
    assertNotNull("words null", words);
    assertEquals(10, words.size());
    // System.out.println("Words: " + words);
    Map<Integer,DateFormat> integerDateFormatMap = model.getDateMap();
    assertNotNull("date format null", integerDateFormatMap);
    assertEquals(1, integerDateFormatMap.size());
    model = new MapBackedARFFModel(model.getWords(), model.getWordCount(), model.getNominalMap());
    iterable = new ARFFVectorIterable(NON_NUMERIC_ARFF2, model);
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
  
  private static final String SAMPLE_DENSE_ARFF = "   % Comments\n" + "   % \n" + "   % Comments go here"
                                                  + "   % \n" + "   @RELATION Mahout\n" + '\n'
                                                  + "   @ATTRIBUTE foo  NUMERIC\n"
                                                  + "   @ATTRIBUTE bar   NUMERIC\n"
                                                  + "   @ATTRIBUTE hockey  NUMERIC\n"
                                                  + "   @ATTRIBUTE football   NUMERIC\n" + "  \n" + '\n'
                                                  + '\n' + "   @DATA\n" + "   23.1,3.23,1.2,0.2\n"
                                                  + "   2.9,3.0,1.2,0.2\n" + "   2.7,3.2,1.3,0.2\n"
                                                  + "   2.6,3.1,1.23,0.2\n" + "   23.0,3.6,1.2,0.2\n"
                                                  + "   23.2,3.9,1.7,0.2\n" + "   2.6,3.2,1.2,0.3\n"
                                                  + "   23.0,3.2,1.23,0.2\n" + "   2.2,2.9,1.2,0.2\n"
                                                  + "   2.9,3.1,1.23,0.1\n";
  
  private static final String SAMPLE_SPARSE_ARFF = "   % Comments\n" + "   % \n" + "   % Comments go here"
                                                   + "   % \n" + "   @RELATION Mahout\n" + '\n'
                                                   + "   @ATTRIBUTE foo  NUMERIC\n"
                                                   + "   @ATTRIBUTE bar   NUMERIC\n"
                                                   + "   @ATTRIBUTE hockey  NUMERIC\n"
                                                   + "   @ATTRIBUTE football   NUMERIC\n"
                                                   + "   @ATTRIBUTE tennis   NUMERIC\n" + "  \n" + '\n'
                                                   + '\n' + "   @DATA\n" + "   {1 23.1,2 3.23,3 1.2,4 0.2}\n"
                                                   + "   {0 2.9}\n" + "   {0 2.7,2 3.2,3 1.3,4 0.2}\n"
                                                   + "   {1 2.6,2 3.1,3 1.23,4 0.2}\n"
                                                   + "   {1 23.0,2 3.6,3 1.2,4 0.2}\n"
                                                   + "   {0 23.2,1 3.9,3 1.7,4 0.2}\n"
                                                   + "   {0 2.6,1 3.2,2 1.2,4 0.3}\n"
                                                   + "   {1 23.0,2 3.2,3 1.23}\n"
                                                   + "   {1 2.2,2 2.94,3 0.2}\n" + "   {1 2.9,2 3.1}\n";
  
  private static final String NON_NUMERIC_ARFF = "   % Comments\n" + "   % \n" + "   % Comments go here"
                                                 + "   % \n" + "   @RELATION Mahout\n" + '\n'
                                                 + "   @ATTRIBUTE junk  NUMERIC\n"
                                                 + "   @ATTRIBUTE foo  NUMERIC\n"
                                                 + "   @ATTRIBUTE bar   {c,d}\n"
                                                 + "   @ATTRIBUTE hockey  string\n"
                                                 + "   @ATTRIBUTE football   date \"yyyy-MM-dd\"\n" + "  \n"
                                                 + '\n' + '\n' + "   @DATA\n"
                                                 + "   {2 c,3 gretzky,4 1973-10-23}\n"
                                                 + "   {1 2.9,2 d,3 orr,4 1973-11-23}\n"
                                                 + "   {2 c,3 bossy,4 1981-10-23}\n"
                                                 + "   {1 2.6,2 c,3 lefleur,4 1989-10-23}\n"
                                                 + "   {3 esposito,4 1973-04-23}\n"
                                                 + "   {1 23.2,2 d,3 chelios,4 1999-2-23}\n"
                                                 + "   {3 richard,4 1973-10-12}\n"
                                                 + "   {3 howe,4 1983-06-23}\n"
                                                 + "   {0 2.2,2 d,3 messier,4 2008-11-23}\n"
                                                 + "   {2 c,3 roy,4 1973-10-13}\n";
  
  private static final String NON_NUMERIC_ARFF2 = "   % Comments\n" + "   % \n" + "   % Comments go here"
                                                  + "   % \n" + "   @RELATION Mahout\n" + '\n'
                                                  + "   @ATTRIBUTE junk  NUMERIC\n"
                                                  + "   @ATTRIBUTE foo  NUMERIC\n"
                                                  + "   @ATTRIBUTE test   {f,z}\n"
                                                  + "   @ATTRIBUTE hockey  string\n"
                                                  + "   @ATTRIBUTE football   date \"yyyy-MM-dd\"\n" + "  \n"
                                                  + '\n' + '\n' + "   @DATA\n"
                                                  + "   {2 f,3 gretzky,4 1973-10-23}\n"
                                                  + "   {1 2.9,2 z,3 orr,4 1973-11-23}\n"
                                                  + "   {2 f,3 bossy,4 1981-10-23}\n"
                                                  + "   {1 2.6,2 f,3 lefleur,4 1989-10-23}\n"
                                                  + "   {3 esposito,4 1973-04-23}\n"
                                                  + "   {1 23.2,2 z,3 chelios,4 1999-2-23}\n"
                                                  + "   {3 richard,4 1973-10-12}\n"
                                                  + "   {3 howe,4 1983-06-23}\n"
                                                  + "   {0 2.2,2 f,3 messier,4 2008-11-23}\n"
                                                  + "   {2 f,3 roy,4 1973-10-13}\n";
  
  private static final String DATE_ARFF = "   % Comments\n"
                                          + "   % \n"
                                          + "   % Comments go here"
                                          + "   % \n"
                                          + "   @RELATION MahoutDateTest\n"
                                          + '\n'
                                          + "   @ATTRIBUTE junk  NUMERIC\n"
                                          + "   @ATTRIBUTE date1   \n"
                                          + "   @ATTRIBUTE date2   date \"yyyy.MM.dd G 'at' HH:mm:ss z\" \n"
                                          + "   @ATTRIBUTE date3   date \"EEE, MMM d, ''yy\" \n"
                                          + "   @ATTRIBUTE date4   date \"K:mm a, z\" \n"
                                          + "   @ATTRIBUTE date5   date \"yyyyy.MMMMM.dd GGG hh:mm aaa\" \n"
                                          + "   @ATTRIBUTE date6   date \"EEE, d MMM yyyy HH:mm:ss Z\" \n"
                                          + "  \n"
                                          + '\n'
                                          + '\n'
                                          + "   @DATA\n"
                                          + "   {0 1,1 \"2001-07-04T12:08:56\",2 \"2001.07.04 AD at 12:08:56 PDT\",3 \"Wed, Jul 4, '01,4 0:08 PM, PDT\",4 \"0:08 PM, PDT\", 5 \"02001.July.04 AD 12:08 PM\" ,6 \"Wed, 4 Jul 2001 12:08:56 -0700\"  }\n"
                                          + "   {0 2,1 \"2001-08-04T12:09:56\",2 \"2011.07.04 AD at 12:08:56 PDT\",3 \"Mon, Jul 4, '11,4 0:08 PM, PDT\",4 \"0:08 PM, PDT\", 5 \"02001.July.14 AD 12:08 PM\" ,6 \"Mon, 4 Jul 2011 12:08:56 -0700\"  }\n";
}
