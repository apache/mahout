/**
 * 
 */
package org.apache.mahout.df.data;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.mahout.df.data.Dataset.Attribute;

/**
 * Contains various methods that deal with descriptor strings
 */
public class DescriptorUtils {
  private DescriptorUtils() {
  }

  /**
   * Parses a descriptor string and generates the corresponding array of Attributes
   * 
   * @param descriptor
   * @return
   * @throws DescriptorException if a bad token is encountered
   */
  public static Attribute[] parseDescriptor(String descriptor) throws DescriptorException {
    StringTokenizer tokenizer = new StringTokenizer(descriptor);
    Attribute[] attributes = new Attribute[tokenizer.countTokens()];
  
    for (int attr = 0; attr < attributes.length; attr++) {
      String token = tokenizer.nextToken().toUpperCase();
      if ("I".equals(token))
        attributes[attr] = Attribute.IGNORED;
      else if ("N".equals(token))
        attributes[attr] = Attribute.NUMERICAL;
      else if ("C".equals(token))
        attributes[attr] = Attribute.CATEGORICAL;
      else if ("L".equals(token)) {
        attributes[attr] = Attribute.LABEL;
      } else
        throw new DescriptorException("Bad Token : " + token);
    }
  
    return attributes;
  }

  /**
   * Generates a valid descriptor string from a user-friendly representation.<br>
   * for example "3 N I N N 2 C L 5 I" generates "N N N I N N C C L I I I I I".<br>
   * this useful when describing datasets with a large number of attributes
   * @param description
   * @return
   * @throws DescriptorException
   */
  public static String generateDescriptor(String description) throws DescriptorException {
    StringTokenizer tokenizer = new StringTokenizer(description, " ");
    List<String> tokens = new ArrayList<String>();
    
    while (tokenizer.hasMoreTokens()) {
      tokens.add(tokenizer.nextToken());
    }
    
    return generateDescriptor(tokens);
  }

  /**
   * Generates a valid descriptor string from a list of tokens
   * @param tokens
   * @return
   * @throws DescriptorException
   */
  public static String generateDescriptor(List<String> tokens) throws DescriptorException {
    StringBuilder descriptor = new StringBuilder();
    
    int multiplicator = 0;

    for (String token : tokens) {
      try {
        // try to parse an integer
        int number = Integer.parseInt(token);

        if (number <= 0) {
          throw new DescriptorException("Multiplicator ("+number+") must be > 0");
        }
        if (multiplicator > 0) {
          throw new DescriptorException("A multiplicator cannot be followed by another multiplicator");
        }
        
        multiplicator = number;
      } catch (NumberFormatException e) {
        // token is not a number
        if (multiplicator == 0) {
          multiplicator = 1;
        }
        
        for (int index=0;index<multiplicator; index++) {
          descriptor.append(token).append(' ');
        }
        
        multiplicator = 0;
      }
    }
    
    return descriptor.toString().trim();
  }
}
