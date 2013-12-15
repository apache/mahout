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

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import org.apache.mahout.math.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.Charset;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Iterator;
import java.util.Locale;

/**
 * Read in ARFF (http://www.cs.waikato.ac.nz/~ml/weka/arff.html) and create {@link Vector}s
 * <p/>
 * Attribute type handling:
 * <ul>
 * <li>Numeric -> As is</li>
 * <li>Nominal -> ordinal(value) i.e. @attribute lumber {'\'(-inf-0.5]\'','\'(0.5-inf)\''}
 * will convert -inf-0.5 -> 0, and 0.5-inf -> 1</li>
 * <li>Dates -> Convert to time as a long</li>
 * <li>Strings -> Create a map of String -> long</li>
 * </ul>
 * NOTE: This class does not set the label bindings on every vector.  If you want the label
 * bindings, call {@link MapBackedARFFModel#getLabelBindings()}, as they are the same for every vector.
 */
public class ARFFVectorIterable implements Iterable<Vector> {

  private final BufferedReader buff;
  private final ARFFModel model;

  public ARFFVectorIterable(File file, ARFFModel model) throws IOException {
    this(file, Charsets.UTF_8, model);
  }

  public ARFFVectorIterable(File file, Charset encoding, ARFFModel model) throws IOException {
    this(Files.newReader(file, encoding), model);
  }

  public ARFFVectorIterable(String arff, ARFFModel model) throws IOException {
    this(new StringReader(arff), model);
  }

  public ARFFVectorIterable(Reader reader, ARFFModel model) throws IOException {
    if (reader instanceof BufferedReader) {
      buff = (BufferedReader) reader;
    } else {
      buff = new BufferedReader(reader);
    }
    //grab the attributes, then start the iterator at the first line of data
    this.model = model;

    int labelNumber = 0;
    String line;
    while ((line = buff.readLine()) != null) {
      line = line.trim();
      if (!line.startsWith(ARFFModel.ARFF_COMMENT) && !line.isEmpty()) {
        Integer labelNumInt = labelNumber;
        String[] lineParts = line.split("[\\s\\t]+", 2);

        // is it a relation name?
        if (lineParts[0].equalsIgnoreCase(ARFFModel.RELATION)) {
          model.setRelation(ARFFType.removeQuotes(lineParts[1]));
        }
        // or an attribute
        else if (lineParts[0].equalsIgnoreCase(ARFFModel.ATTRIBUTE)) {
          String label;
          ARFFType type;

          // split the name of the attribute and its description
          String[] attrParts = lineParts[1].split("[\\s\\t]+", 2);
          if (attrParts.length < 2)
            throw new UnsupportedOperationException("No type for attribute found: " + lineParts[1]);

          // label is attribute name
          label = ARFFType.removeQuotes(attrParts[0].toLowerCase());
          if (attrParts[1].equalsIgnoreCase(ARFFType.NUMERIC.getIndicator())) {
            type = ARFFType.NUMERIC;
          } else if (attrParts[1].equalsIgnoreCase(ARFFType.INTEGER.getIndicator())) {
            type = ARFFType.INTEGER;
          } else if (attrParts[1].equalsIgnoreCase(ARFFType.REAL.getIndicator())) {
            type = ARFFType.REAL;
          } else if (attrParts[1].equalsIgnoreCase(ARFFType.STRING.getIndicator())) {
            type = ARFFType.STRING;
          } else if (attrParts[1].toLowerCase().startsWith(ARFFType.NOMINAL.getIndicator())) {
            type = ARFFType.NOMINAL;
            // nominal example:
            // @ATTRIBUTE class        {Iris-setosa,'Iris versicolor',Iris-virginica}
            String[] classes = ARFFIterator.splitCSV(attrParts[1].substring(1, attrParts[1].length() - 1));
            for (int i = 0; i < classes.length; i++) {
              model.addNominal(label, ARFFType.removeQuotes(classes[i]), i + 1);
            }
          } else if (attrParts[1].toLowerCase().startsWith(ARFFType.DATE.getIndicator())) {
            type = ARFFType.DATE;
            //TODO: DateFormatter map
            DateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.ENGLISH);
            String formStr = attrParts[1].substring(ARFFType.DATE.getIndicator().length()).trim();
            if (!formStr.isEmpty()) {
              if (formStr.startsWith("\"")) {
                formStr = formStr.substring(1, formStr.length() - 1);
              }
              format = new SimpleDateFormat(formStr, Locale.ENGLISH);
            }
            model.addDateFormat(labelNumInt, format);
            //@attribute <name> date [<date-format>]
          } else {
            throw new UnsupportedOperationException("Invalid attribute: " + attrParts[1]);
          }
          model.addLabel(label, labelNumInt);
          model.addType(labelNumInt, type);
          labelNumber++;
        } else if (lineParts[0].equalsIgnoreCase(ARFFModel.DATA)) {
          break; //skip it
        }
      }
    }

  }

  @Override
  public Iterator<Vector> iterator() {
    return new ARFFIterator(buff, model);
  }

  /**
   * Returns info about the ARFF content that was parsed.
   *
   * @return the model
   */
  public ARFFModel getModel() {
    return model;
  }
}
