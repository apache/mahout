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

package org.apache.mahout.ga.watchmaker.cd.tool;

import junit.framework.TestCase;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.ga.watchmaker.cd.tool.DescriptionUtils.Range;
import org.uncommons.maths.random.MersenneTwisterRNG;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

public class CDInfosToolTest extends TestCase {

  Random rng;

  @Override
  protected void setUp() {
    rng = new MersenneTwisterRNG();
  }

  private Descriptors randomDescriptors(int nbattributes, double numRate, double catRate) {
    char[] descriptors = new char[nbattributes];
    double rnd;
    for (int index = 0; index < nbattributes; index++) {
      rnd = rng.nextDouble();
      if (rnd < numRate) {
        // numerical attribute
        descriptors[index] = 'N';
      } else if (rnd < (numRate + catRate)) {
        // categorical attribute
        descriptors[index] = 'C';
      } else {
        // ignored attribute
        descriptors[index] = 'I';
      }
    }

    return new Descriptors(descriptors);
  }

  private Object[][] randomDescriptions(Descriptors descriptors) {
    int nbattrs = descriptors.size();
    Object[][] descriptions = new Object[nbattrs][];

    for (int index = 0; index < nbattrs; index++) {
      if (descriptors.isNumerical(index)) {
        // numerical attribute

        // srowen: I 'fixed' this to not use Double.{MAX,MIN}_VALUE since
        // it does not seem like that has the desired effect
        double min = rng.nextDouble() * ((long) Integer.MAX_VALUE - Integer.MIN_VALUE) + Integer.MIN_VALUE;
        double max = rng.nextDouble() * (Integer.MAX_VALUE - min) + min;

        descriptions[index] = new Double[] { min, max };
      } else if (descriptors.isNominal(index)) {
        // categorical attribute
        int nbvalues = rng.nextInt(50) + 1;
        descriptions[index] = new Object[nbvalues];
        for (int vindex = 0; vindex < nbvalues; vindex++) {
          descriptions[index][vindex] = "val_" + index + "_" + vindex;
        }
      }
    }

    return descriptions;
  }

  private void randomDataset(FileSystem fs, Path input, Descriptors descriptors,
      Object[][] descriptions) throws IOException {
    int nbfiles = rng.nextInt(20) + 1;
    FSDataOutputStream out;
    BufferedWriter writer;

    for (int floop = 0; floop < nbfiles; floop++) {
      out = fs.create(new Path(input, "file." + floop));
      writer = new BufferedWriter(new OutputStreamWriter(out));

      int nblines = rng.nextInt(200) + 1;
      for (int line = 0; line < nblines; line++) {
        writer.write(randomLine(descriptors, descriptions));
        writer.newLine();
      }

      writer.close();
    }
  }

  private String randomLine(Descriptors descriptors, Object[][] descriptions) {
    StringBuffer buffer = new StringBuffer();

    for (int index = 0; index < descriptors.size(); index++) {
      if (descriptors.isNumerical(index)) {
        // numerical attribute
        double min = (Double) descriptions[index][0];
        double max = (Double) descriptions[index][1];
        double value = rng.nextDouble() * (max - min) + min;

        buffer.append(value);
      } else if (descriptors.isNominal(index)) {
        // categorical attribute
        int nbvalues = descriptions[index].length;
        int vindex = rng.nextInt(nbvalues);

        buffer.append(descriptions[index][vindex]);
      } else {
        // ignored attribute (any value is correct)
        buffer.append('I');
      }

      if (index < descriptors.size() - 1) {
        buffer.append(',');
      }
    }

    return buffer.toString();
  }

  private int nbNonIgnored(Descriptors descriptors) {
    int nbattrs = 0;
    for (int index = 0; index < descriptors.size(); index++) {
      if (!descriptors.isIgnored(index))
        nbattrs++;
    }
    
    return nbattrs;
  }

  public void testGatherInfos() throws Exception {
    int maxattr = 100; // max number of attributes
    int nbattrs = rng.nextInt(maxattr) + 1;

    // random descriptors
    double numRate = rng.nextDouble();
    double catRate = rng.nextDouble() * (1.0 - numRate);
    Descriptors descriptors = randomDescriptors(nbattrs, numRate, catRate);

    // random descriptions
    Object[][] descriptions = randomDescriptions(descriptors);

    // random dataset
    Path inpath = new Path("input");
    FileSystem fs = FileSystem.get(inpath.toUri(), new Configuration());
    if (fs.exists(inpath)) {
      fs.delete(inpath, true);
    }

    randomDataset(fs, inpath, descriptors, descriptions);

    // Start the tool
    List<String> result = new ArrayList<String>();
    int rindex=0;
    CDInfosTool.gatherInfos(descriptors, inpath, result);

    // check the results
    Collection<String> target = new ArrayList<String>();

    assertEquals(nbNonIgnored(descriptors), result.size());
    for (int index = 0; index < nbattrs; index++) {
      if (descriptors.isIgnored(index)) {
        continue;
      } 

      String description = result.get(rindex++);

      if (descriptors.isNumerical(index)) {
        // numerical attribute
        double min = (Double) descriptions[index][0];
        double max = (Double) descriptions[index][1];
        Range range = DescriptionUtils.extractNumericalRange(description);

        assertTrue("bad min value for attribute (" + index + ")",
            min <= range.min);
        assertTrue("bad max value for attribute (" + index + ")",
            max >= range.max);
      } else if (descriptors.isNominal(index)) {
        // categorical attribute
        Object[] values = descriptions[index];
        target.clear();
        DescriptionUtils.extractNominalValues(description, target);

        assertEquals(values.length, target.size());
        assertTrue(target.containsAll(Arrays.asList(values)));
      }
    }
  }

}
