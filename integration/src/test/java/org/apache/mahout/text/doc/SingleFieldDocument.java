package org.apache.mahout.text.doc;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;

/**
 * Used for testing lucene2seq
 */
public class SingleFieldDocument {

  public static final String ID_FIELD = "idField";
  public static final String FIELD = "field";

  private String id;
  private String field;

  public SingleFieldDocument(String id, String field) {
    this.id = id;
    this.field = field;
  }

  public String getId() {
    return id;
  }

  public String getField() {
    return field;
  }

  public Document asLuceneDocument() {
    Document document = new Document();

    Field idField = new StringField(ID_FIELD, getId(), Field.Store.YES);
    Field field = new TextField(FIELD, getField(), Field.Store.YES);

    document.add(idField);
    document.add(field);

    return document;
  }
}
