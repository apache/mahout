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

import com.google.common.collect.ImmutableList;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;

import static org.junit.Assert.assertEquals;


public class LogisticModelParametersTest {
  @Test
  public void testSaveTo() throws IOException {
    LogisticModelParameters lmp = new LogisticModelParameters();
    lmp.setTargetVariable("target");
    lmp.setMaxTargetCategories(3);

    lmp.setTypeMap(ImmutableList.of("x", "y", "z"), ImmutableList.of("n", "word", "numeric"));
    lmp.setUseBias(true);

    lmp.setLambda(123.4);
    lmp.setLearningRate(5.2);

    lmp.setNumFeatures(214);

    lmp.getCsvRecordFactory().firstLine("x,target,q,z,r,y");
    Vector v = new DenseVector(20);
    assertEquals(0, lmp.getCsvRecordFactory().processLine("3,t_1,foo,5,r,cat", v));
    assertEquals(1, lmp.getCsvRecordFactory().processLine("3,t_2,foo,5,r,dog", v));
    assertEquals(2, lmp.getCsvRecordFactory().processLine("3,t_3,foo,5,r,pig", v));
    assertEquals(2, lmp.getCsvRecordFactory().processLine("3,t_4,foo,5,r,pig", v));

    assertEquals(3, lmp.getMaxTargetCategories());
    assertEquals("[t_1, t_2, t_3]", lmp.getCsvRecordFactory().getTargetCategories().toString());

    StringWriter s = new StringWriter();
    lmp.saveTo(s);
    s.close();
    assertEquals("{\"targetVariable\":\"target\",\"typeMap\":{\"z\":\"numeric\",\"y\":\"word\",\"x\":\"n\"},\n" +
            "  \"numFeatures\":214,\"useBias\":true,\"maxTargetCategories\":3,\n" +
            "  \"targetCategories\":[\"t_1\",\"t_2\",\"t_3\"],\"lambda\":123.4,\"learningRate\":5.2}", s.toString().trim());
  }

  @Test
  public void testSaveWithRegression() throws IOException {
    LogisticModelParameters lmp = new LogisticModelParameters();
    lmp.setTargetVariable("target");
    lmp.setMaxTargetCategories(3);

    lmp.setTypeMap(ImmutableList.of("x", "y", "z"), ImmutableList.of("n", "word", "numeric"));
    lmp.setUseBias(true);

    lmp.setLambda(123.4);
    lmp.setLearningRate(5.2);

    lmp.setNumFeatures(214);

    OnlineLogisticRegression lr = lmp.createRegression();
    lr.getBeta().set(0, 4, 5.0);
    lr.getBeta().set(1, 3, 7.0);
    lmp.getCsvRecordFactory().firstLine("x,target,q,z,r,y");
    Vector v = new DenseVector(20);
    assertEquals(0, lmp.getCsvRecordFactory().processLine("3,t_1,foo,5,r,cat", v));
    assertEquals(1, lmp.getCsvRecordFactory().processLine("3,t_2,foo,5,r,dog", v));
    assertEquals(2, lmp.getCsvRecordFactory().processLine("3,t_3,foo,5,r,pig", v));
    assertEquals(2, lmp.getCsvRecordFactory().processLine("3,t_4,foo,5,r,pig", v));

    assertEquals(3, lmp.getMaxTargetCategories());
    assertEquals("[t_1, t_2, t_3]", lmp.getCsvRecordFactory().getTargetCategories().toString());

    StringWriter s = new StringWriter();
    lmp.saveTo(s);
    s.close();
    assertEquals("{\"targetVariable\":\"target\",\"typeMap\":{\"z\":\"numeric\",\"y\":\"word\",\"x\":\"n\"},\n" +
            "  \"numFeatures\":214,\"useBias\":true,\"maxTargetCategories\":3,\n" +
            "  \"targetCategories\":[\"t_1\",\"t_2\",\"t_3\"],\"lambda\":123.4,\"learningRate\":5.2,\n" +
            "  \"lr\":{\"beta\":{\"rows\":2,\"cols\":214,\"data\":[[0.0,0.0,0.0,0.0,5.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[\n" +
            "          0.0,0.0,0.0,7.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
            "          0.0,0.0,0.0,0.0,0.0,0.0]]},\"numCategories\":3,\"step\":1,\"mu_0\":5.2,\n" +
            "    \"decayFactor\":0.999,\"stepOffset\":10,\"decayExponent\":-0.5,\"lambda\":\n" +
            "    123.4,\"sealed\":true}}", s.toString().trim());
  }

  @Test
  public void testLoadFrom() {
    LogisticModelParameters lmp = LogisticModelParameters.loadFrom(new StringReader(
            "{\"targetVariable\":\"target\",\"typeMap\":{\"z\":\"numeric\",\"y\":\"word\",\"x\":\"n\"},\n" +
                    "  \"numFeatures\":214,\"useBias\":true,\"maxTargetCategories\":3,\n" +
                    "  \"targetCategories\":[\"t_1\",\"t_2\",\"t_3\"],\"lambda\":123.4,\"learningRate\":5.2,\n" +
                    "  \"lr\":{\"beta\":{\"rows\":2,\"cols\":214,\"data\":[[0.0,0.0,0.0,0.0,5.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[\n" +
                    "          0.0,0.0,0.0,7.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\n" +
                    "          0.0,0.0,0.0,0.0,0.0,0.0]]},\"numCategories\":3,\"step\":1,\"mu_0\":5.2,\n" +
                    "    \"decayFactor\":0.999,\"stepOffset\":10,\"decayExponent\":-0.5,\"lambda\":\n" +
                    "    123.4,\"sealed\":true}}"));

    assertEquals(5.0, lmp.createRegression().getBeta().get(0, 4), 0);
    assertEquals(7.0, lmp.createRegression().getBeta().get(1, 3), 0);

    assertEquals(123.4, lmp.getLambda(), 0.0);
    assertEquals(true, lmp.useBias());
    assertEquals(5.2, lmp.getLearningRate(), 0.0);
    assertEquals(214, lmp.getNumFeatures());

    lmp.getCsvRecordFactory().firstLine("x,target,q,z,r,y");
    Vector v = new DenseVector(20);
    assertEquals(2, lmp.getCsvRecordFactory().processLine("3,t_3,foo,5,r,pig", v));
    assertEquals(1, lmp.getCsvRecordFactory().processLine("3,t_2,foo,5,r,dog", v));
    assertEquals(2, lmp.getCsvRecordFactory().processLine("3,t_4,foo,5,r,pig", v));
    assertEquals(0, lmp.getCsvRecordFactory().processLine("3,t_1,foo,5,r,cat", v));

    assertEquals(3, lmp.getMaxTargetCategories());
    assertEquals("[t_1, t_2, t_3]", lmp.getCsvRecordFactory().getTargetCategories().toString());

    assertEquals(214, lmp.createRegression().getBeta().numCols());
    assertEquals(2, lmp.createRegression().getBeta().numRows());
  }
}
