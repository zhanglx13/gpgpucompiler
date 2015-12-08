package cetus.hir;

import java.io.*;
import java.lang.reflect.*;

/**
 * AnnotationStatement represents an empty statement 
 * to hold Annotations. It is preserved by the IR.
 */

public class AnnotationStatement extends Statement
{
  private static Method class_print_method;

  private Annotation annot;

  /**
    * each AnnotationStatement can set a Statement to which it is attached
    * standalone directives, such as OpenMP barrier/flush and comments,
    * attached_stmt should be null
    */
  private Statement attached_stmt;

  static
  {
    Class[] params = new Class[2];

    try {
      params[0] = AnnotationStatement.class;
      params[1] = OutputStream.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  /**
   * Create a new break statement.
   */
  public AnnotationStatement(Annotation iannot)
  {
    object_print_method = class_print_method;
		annot = iannot;
  }

  /**
   * Prints a statement to a stream.
   *
   * @param stmt The statement to print.
   * @param stream The stream on which to print the statement.
   */
  public static void defaultPrint(AnnotationStatement stmt, OutputStream stream)
  {
    PrintStream p = new PrintStream(stream);

		p.print(stmt.annot.toString());
  }

	public String toString()
	{
		StringBuilder str = new StringBuilder(annot.toString());

		return str.toString();
	}

  /**
   * Returns the declaration part of the statement.
   *
   * @return the declaration part of the statement.
   */
  public Annotation getAnnotation()
  {
    return annot;
  }

  public void attachStatement(Statement stmt)
  {
    attached_stmt = stmt;
  }

  public Statement getStatement()
  {
    return attached_stmt;
  }

  /**
   * Overrides the class print method, so that all subsequently
   * created objects will use the supplied method.
   *
   * @param m The new print method.
   */
  static public void setClassPrintMethod(Method m)
  {
    class_print_method = m;
  }
}
