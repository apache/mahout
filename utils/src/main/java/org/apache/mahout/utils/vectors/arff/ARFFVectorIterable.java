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

import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.common.IOUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.text.DateFormat;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.regex.Pattern;

/**
 * Read in ARFF (http://www.cs.waikato.ac.nz/~ml/weka/arff.html) and create {@link Vector}s
 * <p/>
 * Attribute type handling:
 * <ul>
 * <li>Numeric -> As is</li>
 * <li>Nominal -> ordinal(value) i.e. @attribute lumber {'\'(-inf-0.5]\'','\'(0.5-inf)\''}
 *  will convert -inf-0.5 -> 0, and 0.5-inf -> 1</li>
 * <li>Dates -> Convert to time as a long</li>
 * <li>Strings -> Create a map of String -> long</li>
 * </ul>
 * NOTE: This class does not set the label bindings on every vector.  If you want the label
 * bindings, call {@link MapBackedARFFModel#getLabelBindings()}, as they are the same for every vector.
 */
public class ARFFVectorIterable implements Iterable<Vector> {

  private static final Pattern COMMA_PATTERN = Pattern.compile(",");
  private static final Pattern SPACE_PATTERN = Pattern.compile(" ");

  private final BufferedReader buff;
  private final ARFFModel model;

  public ARFFVectorIterable(File file, ARFFModel model) throws IOException {
    this(new FileReader(file), model);
  }

  public ARFFVectorIterable(File file, Charset encoding, ARFFModel model) throws IOException {
    this(new InputStreamReader(new FileInputStream(file), encoding), model);
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
    //boolean inData = false; // TODO not used?
    while ((line = buff.readLine()) != null) {
      line = line.trim();
      String lower = line.toLowerCase();
      Integer labelNumInt = labelNumber;
      if (lower.startsWith(ARFFModel.ARFF_COMMENT)) {
        continue;
      } else if (lower.startsWith(ARFFModel.RELATION)) {
        model.setRelation(line.substring(ARFFModel.RELATION.length()).trim());
      } else if (lower.startsWith(ARFFModel.ATTRIBUTE)) {
        String label;
        ARFFType type;
        if (lower.contains(ARFFType.NUMERIC.getIndicator())) {
          label = ARFFType.NUMERIC.getLabel(lower);
          type = ARFFType.NUMERIC;
        } else if (lower.contains(ARFFType.STRING.getIndicator())) {
          label = ARFFType.STRING.getLabel(lower);
          type = ARFFType.STRING;
          //TODO: create a map so we know which

        } else if (lower.contains(ARFFType.NOMINAL.getIndicator())) {
          label = ARFFType.NOMINAL.getLabel(lower);
          type = ARFFType.NOMINAL;
          //@ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}
          int classIdx = lower.indexOf(ARFFType.NOMINAL.getIndicator());
          String [] classes = COMMA_PATTERN.split(line.substring(classIdx + 1, line.length() - 1));
          for (int i = 0; i < classes.length; i++) {
            model.addNominal(label, classes[i].trim(), i);
          }

        } else if (lower.contains(ARFFType.DATE.getIndicator())) {
          label = ARFFType.DATE.getLabel(lower);
          type = ARFFType.DATE;
          //TODO: DateFormatter map
          DateFormat format = ARFFModel.DEFAULT_DATE_FORMAT;
          int idx = lower.indexOf(ARFFType.DATE.getIndicator());
          String[] split = SPACE_PATTERN.split(line);
          if (split.length >= 4) {//we have a date format
            String formStr = line.substring(idx + ARFFType.DATE.getIndicator().length()).trim();
            if (formStr.startsWith("\"")) {
              formStr = formStr.substring(1, formStr.length() - 1);
            }
            format = new SimpleDateFormat(formStr);
          }
          model.addDateFormat(labelNumInt, format);
          //@attribute <name> date [<date-format>]
        } else {
          throw new UnsupportedOperationException("Invalid attribute: " + line);
        }
        model.addLabel(label, labelNumInt);
        model.addType(labelNumInt, type);
        labelNumber++;
      } else if (lower.startsWith(ARFFModel.DATA)) {
        //inData = true;
        break;//skip it
      }
    }

  }


  @Override
  public Iterator<Vector> iterator() {
    return new ARFFIterator();
  }

  private class ARFFIterator implements Iterator<Vector> {

    private String line;

    private ARFFIterator() {
      goToNext();
    }

    private void goToNext() {
      line = null;
      try {
        while ((line = buff.readLine()) != null) {
          line = line.trim();
          if (line.length() > 0 && line.startsWith(ARFFModel.ARFF_COMMENT) == false) {
            break;
          }
        }
      } catch (IOException e) {
        line = null;
      }
      if (line == null) {
        IOUtils.quietClose(buff);
      }
    }

    @Override
    public boolean hasNext() {
      return line != null;
    }

    @Override
    public Vector next() {
      if (line == null) {
        throw new NoSuchElementException();
      }
      Vector result;
      if (line.startsWith(ARFFModel.ARFF_SPARSE)) {
        line = line.substring(1, line.length() - 1);
        String[] splits = COMMA_PATTERN.split(line);
        result = new SparseVector(model.getLabelSize());
        for (String split : splits) {
          String[] data = SPACE_PATTERN.split(split); // first is index, second is
          int idx = Integer.parseInt(data[0]);
          result.setQuick(idx, model.getValue(data[1], idx));
        }
      } else {
        result = new DenseVector(model.getLabelSize());
        String[] splits = COMMA_PATTERN.split(line);
        for (int i = 0; i < splits.length; i++) {
          result.setQuick(i, model.getValue(splits[i], i));
        }
      }
      //result.setLabelBindings(labelBindings);
      goToNext();
      return result;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException("remove not supported");
    }
  }

  /**
   * Returns info about the ARFF content that was parsed.
   * @return the model
   */
  public ARFFModel getModel() {
    return model;
  }
}
