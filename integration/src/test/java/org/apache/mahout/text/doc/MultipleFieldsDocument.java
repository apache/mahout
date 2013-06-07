package org.apache.mahout.text.doc;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;

/**
 * Used for testing lucene2seq
 */
public class MultipleFieldsDocument extends SingleFieldDocument {

  public static final String FIELD1 = "field1";
  public static final String FIELD2 = "field2";

  private String field1;
  private String field2;

  public MultipleFieldsDocument(String id, String field, String field1, String field2) {
    super(id, field);
    this.field1 = field1;
    this.field2 = field2;
  }

  public String getField1() {
    return field1;
  }

  public String getField2() {
    return field2;
  }

  @Override
  public Document asLuceneDocument() {
    Document document = super.asLuceneDocument();

    Field field1 = new Field(FIELD1, this.field1, Field.Store.YES, Field.Index.ANALYZED);
    Field field2 = new Field(FIELD2, this.field2, Field.Store.YES, Field.Index.ANALYZED);

    document.add(field1);
    document.add(field2);

    return document;
  }
}
