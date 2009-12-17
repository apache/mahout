package org.apache.mahout.math.matrix.objectalgo;

/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/

import org.apache.mahout.math.matrix.ObjectMatrix1D;
import org.apache.mahout.math.matrix.ObjectMatrix2D;
import org.apache.mahout.math.matrix.ObjectMatrix3D;
import org.apache.mahout.math.matrix.impl.AbstractFormatter;
import org.apache.mahout.math.matrix.impl.AbstractMatrix1D;
import org.apache.mahout.math.matrix.impl.AbstractMatrix2D;
import org.apache.mahout.math.matrix.impl.Former;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Formatter extends AbstractFormatter {

  /** Constructs and returns a matrix formatter with alignment <tt>LEFT</tt>. */
  public Formatter() {
    this(LEFT);
  }

  /**
   * Constructs and returns a matrix formatter.
   *
   * @param alignment the given alignment used to align a column.
   */
  public Formatter(String alignment) {
    setAlignment(alignment);
  }

  /** Converts a given cell to a String; no alignment considered. */
  @Override
  protected String form(AbstractMatrix1D matrix, int index, Former formatter) {
    return this.form((ObjectMatrix1D) matrix, index, formatter);
  }

  /** Converts a given cell to a String; no alignment considered. */
  protected String form(ObjectMatrix1D matrix, int index, Former formatter) {
    Object value = matrix.get(index);
    if (value == null) {
      return "";
    }
    return String.valueOf(value);
  }

  /** Returns a string representations of all cells; no alignment considered. */
  @Override
  protected String[][] format(AbstractMatrix2D matrix) {
    return this.format((ObjectMatrix2D) matrix);
  }

  /** Returns a string representations of all cells; no alignment considered. */
  protected String[][] format(ObjectMatrix2D matrix) {
    String[][] strings = new String[matrix.rows()][matrix.columns()];
    for (int row = matrix.rows(); --row >= 0;) {
      strings[row] = formatRow(matrix.viewRow(row));
    }
    return strings;
  }

  /**
   * Returns a string <tt>s</tt> such that <tt>Object[] m = s</tt> is a legal Java statement.
   *
   * @param matrix the matrix to format.
   */
  public String toSourceCode(ObjectMatrix1D matrix) {
    Formatter copy = (Formatter) this.clone();
    copy.setPrintShape(false);
    copy.setColumnSeparator(", ");
    String lead = "{";
    String trail = "};";
    return lead + copy.toString(matrix) + trail;
  }

  /**
   * Returns a string <tt>s</tt> such that <tt>Object[] m = s</tt> is a legal Java statement.
   *
   * @param matrix the matrix to format.
   */
  public String toSourceCode(ObjectMatrix2D matrix) {
    Formatter copy = (Formatter) this.clone();
    String b3 = blanks(3);
    copy.setPrintShape(false);
    copy.setColumnSeparator(", ");
    copy.setRowSeparator("},\n" + b3 + '{');
    String lead = "{\n" + b3 + '{';
    String trail = "}\n};";
    return lead + copy.toString(matrix) + trail;
  }

  /**
   * Returns a string <tt>s</tt> such that <tt>Object[] m = s</tt> is a legal Java statement.
   *
   * @param matrix the matrix to format.
   */
  public String toSourceCode(ObjectMatrix3D matrix) {
    Formatter copy = (Formatter) this.clone();
    String b3 = blanks(3);
    String b6 = blanks(6);
    copy.setPrintShape(false);
    copy.setColumnSeparator(", ");
    copy.setRowSeparator("},\n" + b6 + '{');
    copy.setSliceSeparator("}\n" + b3 + "},\n" + b3 + "{\n" + b6 + '{');
    String lead = "{\n" + b3 + "{\n" + b6 + '{';
    String trail = "}\n" + b3 + "}\n}";
    return lead + copy.toString(matrix) + trail;
  }

  /**
   * Returns a string representation of the given matrix.
   *
   * @param matrix the matrix to convert.
   */
  @Override
  protected String toString(AbstractMatrix2D matrix) {
    return this.toString((ObjectMatrix2D) matrix);
  }

  /**
   * Returns a string representation of the given matrix.
   *
   * @param matrix the matrix to convert.
   */
  public String toString(ObjectMatrix1D matrix) {
    ObjectMatrix2D easy = matrix.like2D(1, matrix.size());
    easy.viewRow(0).assign(matrix);
    return toString(easy);
  }

  /**
   * Returns a string representation of the given matrix.
   *
   * @param matrix the matrix to convert.
   */
  public String toString(ObjectMatrix2D matrix) {
    return super.toString(matrix);
  }

  /**
   * Returns a string representation of the given matrix.
   *
   * @param matrix the matrix to convert.
   */
  public String toString(ObjectMatrix3D matrix) {
    StringBuilder buf = new StringBuilder();
    boolean oldPrintShape = this.printShape;
    this.printShape = false;
    for (int slice = 0; slice < matrix.slices(); slice++) {
      if (slice != 0) {
        buf.append(sliceSeparator);
      }
      buf.append(toString(matrix.viewSlice(slice)));
    }
    this.printShape = oldPrintShape;
    if (printShape) {
      buf.insert(0, shape(matrix) + '\n');
    }
    return buf.toString();
  }

  /**
   * Returns a string representation of the given matrix with axis as well as rows and columns labeled. Pass
   * <tt>null</tt> to one or more parameters to indicate that the corresponding decoration element shall not appear in
   * the string converted matrix.
   *
   * @param matrix         The matrix to format.
   * @param rowNames       The headers of all rows (to be put to the left of the matrix).
   * @param columnNames    The headers of all columns (to be put to above the matrix).
   * @param rowAxisName    The label of the y-axis.
   * @param columnAxisName The label of the x-axis.
   * @param title          The overall title of the matrix to be formatted.
   * @return the matrix converted to a string.
   */
  public String toTitleString(ObjectMatrix2D matrix, String[] rowNames, String[] columnNames, String rowAxisName,
                              String columnAxisName, String title) {
    if (matrix.size() == 0) {
      return "Empty matrix";
    }
    String oldFormat = this.format;
    this.format = LEFT;

    int rows = matrix.rows();
    int columns = matrix.columns();

    // determine how many rows and columns are needed
    int r = 0;
    r += (columnNames == null ? 0 : 1);
    int c = 0;
    c += (rowNames == null ? 0 : 1);
    c += (rowAxisName == null ? 0 : 1);
    c += (rowNames != null || rowAxisName != null ? 1 : 0);

    int height = r + Math.max(rows, rowAxisName == null ? 0 : rowAxisName.length());
    int width = c + columns;

    // make larger matrix holding original matrix and naming strings
    ObjectMatrix2D titleMatrix = matrix.like(height, width);

    // insert original matrix into larger matrix
    titleMatrix.viewPart(r, c, rows, columns).assign(matrix);

    // insert column axis name in leading row
    if (r > 0) {
      titleMatrix.viewRow(0).viewPart(c, columns).assign(columnNames);
    }

    // insert row axis name in leading column
    if (rowAxisName != null) {
      String[] rowAxisStrings = new String[rowAxisName.length()];
      for (int i = rowAxisName.length(); --i >= 0;) {
        rowAxisStrings[i] = rowAxisName.substring(i, i + 1);
      }
      titleMatrix.viewColumn(0).viewPart(r, rowAxisName.length()).assign(rowAxisStrings);
    }
    // insert row names in next leading columns
    if (rowNames != null) {
      titleMatrix.viewColumn(c - 2).viewPart(r, rows).assign(rowNames);
    }

    // insert vertical "---------" separator line in next leading column
    if (c > 0) {
      titleMatrix.viewColumn(c - 2 + 1).viewPart(0, rows + r).assign("|");
    }

    // convert the large matrix to a string
    boolean oldPrintShape = this.printShape;
    this.printShape = false;
    String str = toString(titleMatrix);
    this.printShape = oldPrintShape;

    // insert horizontal "--------------" separator line
    StringBuilder total = new StringBuilder(str);
    if (columnNames != null) {
      int i = str.indexOf(rowSeparator);
      total.insert(i + 1, repeat('-', i) + rowSeparator);
    } else if (columnAxisName != null) {
      int i = str.indexOf(rowSeparator);
      total.insert(0, repeat('-', i) + rowSeparator);
    }

    // insert line for column axis name
    if (columnAxisName != null) {
      int j = 0;
      if (c > 0) {
        j = str.indexOf('|');
      }
      String s = blanks(j);
      if (c > 0) {
        s += "| ";
      }
      s = s + columnAxisName + '\n';
      total.insert(0, s);
    }

    // insert title
    if (title != null) {
      total.insert(0, title + '\n');
    }

    this.format = oldFormat;

    return total.toString();
  }

  /**
   * Returns a string representation of the given matrix with axis as well as rows and columns labeled. Pass
   * <tt>null</tt> to one or more parameters to indicate that the corresponding decoration element shall not appear in
   * the string converted matrix.
   *
   * @param matrix         The matrix to format.
   * @param sliceNames     The headers of all slices (to be put above each slice).
   * @param rowNames       The headers of all rows (to be put to the left of the matrix).
   * @param columnNames    The headers of all columns (to be put to above the matrix).
   * @param sliceAxisName  The label of the z-axis (to be put above each slice).
   * @param rowAxisName    The label of the y-axis.
   * @param columnAxisName The label of the x-axis.
   * @param title          The overall title of the matrix to be formatted.
   * @return the matrix converted to a string.
   */
  public String toTitleString(ObjectMatrix3D matrix, String[] sliceNames, String[] rowNames, String[] columnNames,
                              String sliceAxisName, String rowAxisName, String columnAxisName, String title) {
    if (matrix.size() == 0) {
      return "Empty matrix";
    }
    StringBuilder buf = new StringBuilder();
    for (int i = 0; i < matrix.slices(); i++) {
      if (i != 0) {
        buf.append(sliceSeparator);
      }
      buf.append(toTitleString(matrix.viewSlice(i), rowNames, columnNames, rowAxisName, columnAxisName,
          title + '\n' + sliceAxisName + '=' + sliceNames[i]));
    }
    return buf.toString();
  }
}
