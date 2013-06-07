package org.apache.mahout.text.doc;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;


/**
 * Document with numeric field.
 */
public class NumericFieldDocument extends SingleFieldDocument {

  public static final String NUMERIC_FIELD = "numeric";

  private int numericField;

  public NumericFieldDocument(String id, String field, int numericField) {
    super(id, field);
    this.numericField = numericField;
  }

  @Override
  public Document asLuceneDocument() {
    Document document = new Document();

    document.add(new StringField(ID_FIELD, getId(), Field.Store.YES));
    document.add(new TextField(FIELD, getField(), Field.Store.YES));
    document.add(new IntField(NUMERIC_FIELD, numericField, Field.Store.YES));

    return document;
  }

  public int getNumericField() {
    return numericField;
  }
}
