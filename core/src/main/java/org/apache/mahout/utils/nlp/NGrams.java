package org.apache.mahout.utils.nlp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

public class NGrams {

  private String line;

  private int gramSize;

  public NGrams(String line, int gramSize) {
    this.line = line;
    this.gramSize = gramSize;
  }

  public Map<String, List<String>> generateNGrams() {
    Map<String, List<String>> returnDocument = new HashMap<String, List<String>>();

    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = new ArrayList<String>();
    String labelName = tokenizer.nextToken();
    List<String> previousN_1Grams = new ArrayList<String>();
    while (tokenizer.hasMoreTokens()) {

      String next_token = tokenizer.nextToken();
      if (previousN_1Grams.size() == gramSize)
        previousN_1Grams.remove(0);

      previousN_1Grams.add(next_token);

      StringBuilder gramBuilder = new StringBuilder();

      for (String gram : previousN_1Grams) {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();
        tokens.add(token);
        gramBuilder.append(' ');
      }
    }
    returnDocument.put(labelName, tokens);
    return returnDocument;
  }

  public List<String> generateNGramsWithoutLabel() {

    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = new ArrayList<String>();

    List<String> previousN_1Grams = new ArrayList<String>();
    while (tokenizer.hasMoreTokens()) {

      String next_token = tokenizer.nextToken();
      if (previousN_1Grams.size() == gramSize)
        previousN_1Grams.remove(0);

      previousN_1Grams.add(next_token);

      StringBuilder gramBuilder = new StringBuilder();

      for (String gram : previousN_1Grams) {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();
        tokens.add(token);
        gramBuilder.append(' ');
      }
    }

    return tokens;
  }
}
