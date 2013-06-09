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

package org.apache.mahout.cf.taste.example.kddcup;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.zip.GZIPOutputStream;


import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.Pair;

/**
 * <p>This class converts a KDD Cup input file into a compressed CSV format. The output format is
 * {@code userID,itemID,score,timestamp}. It can optionally restrict its output to exclude
 * score and/or timestamp.</p>
 *
 * <p>Run as: {@code ToCSV (input file) (output file) [num columns to output]}</p>
 */
public final class ToCSV {

  private ToCSV() {
  }

  public static void main(String[] args) throws Exception {

    File inputFile = new File(args[0]);
    File outputFile = new File(args[1]);
    int columnsToOutput = 4;
    if (args.length >= 3) {
      columnsToOutput = Integer.parseInt(args[2]);
    }

    OutputStream outStream = new GZIPOutputStream(new FileOutputStream(outputFile));
    Writer outWriter = new BufferedWriter(new OutputStreamWriter(outStream, Charsets.UTF_8));

    try {
      for (Pair<PreferenceArray,long[]> user : new DataFileIterable(inputFile)) {
        PreferenceArray prefs = user.getFirst();
        long[] timestamps = user.getSecond();
        for (int i = 0; i < prefs.length(); i++) {
          outWriter.write(String.valueOf(prefs.getUserID(i)));
          outWriter.write(',');
          outWriter.write(String.valueOf(prefs.getItemID(i)));
          if (columnsToOutput > 2) {
            outWriter.write(',');
            outWriter.write(String.valueOf(prefs.getValue(i)));
          }
          if (columnsToOutput > 3) {
            outWriter.write(',');
            outWriter.write(String.valueOf(timestamps[i]));
          }
          outWriter.write('\n');
        }
      }
    } finally {
      Closeables.close(outWriter, false);
    }
  }

}
