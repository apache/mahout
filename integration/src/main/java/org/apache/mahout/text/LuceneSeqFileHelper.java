package org.apache.mahout.text;


import org.apache.hadoop.io.Text;
import org.apache.lucene.document.Document;

import java.util.List;

import static org.apache.commons.lang.StringUtils.isNotBlank;

/**
 *
 *
 **/
class LuceneSeqFileHelper {
  public static final String SEPARATOR_FIELDS = " ";
  public static final int USE_TERM_INFOS = 1;

  public static void populateValues(Document document, Text theValue, List<String> fields) {

    StringBuilder valueBuilder = new StringBuilder();
    for (int i = 0; i < fields.size(); i++) {
      String field = fields.get(i);
      String fieldValue = document.get(field);
      if (isNotBlank(fieldValue)) {
        valueBuilder.append(fieldValue);
        if (i != fields.size() - 1) {
          valueBuilder.append(SEPARATOR_FIELDS);
        }
      }
    }
    theValue.set(nullSafe(valueBuilder.toString()));
  }

  public static String nullSafe(String value) {
    if (value == null) {
      return "";
    } else {
      return value;
    }
  }
}
