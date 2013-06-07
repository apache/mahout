package org.apache.mahout.text.doc;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;

/**
 * Used for testing lucene2seq
 */
public class UnstoredFieldsDocument extends SingleFieldDocument {

  public static final String UNSTORED_FIELD = "unstored";

  public UnstoredFieldsDocument(String id, String field) {
    super(id, field);
  }

  @Override
  public Document asLuceneDocument() {
    Document document = super.asLuceneDocument();

    Field unstoredField = new Field(UNSTORED_FIELD, "", Field.Store.NO, Field.Index.NOT_ANALYZED);

    document.add(unstoredField);

    return document;
  }
}
