package cetus.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

/**
 * Represents a text annotation (comment or pragma) or a list of keyed values.
 * Internally, an annotation is a single string <i>or</i> a map of
 * keys to values.  There is no restriction on the type or content of the
 * keys and values.  Compiler passes are free to use annotations as they see
 * fit, although annotations used by multiple passes should be well-documented.
 *
 * By default, annotations are printed as a multi-line comment. They can also
 * be printed as pragmas.  If the text value of the annotation has been set,
 * the text is printed, otherwise the list of keyed values is printed.
 */

/** ----------------------------------------------------------------
	* if it is a 
	* ----------------------------------------------------------------
	*/

public final class Annotation extends Declaration
{
  private static Method class_print_method;

  /** Useful for passing to setPrintMethod or setClassPrintMethod. */
  public static final Method print_as_comment_method;
  /** Useful for passing to setPrintMethod or setClassPrintMethod. */
  public static final Method print_as_pragma_method;
  /** Useful for passing to setPrintMethod or setClassPrintMethod. */
  public static final Method print_raw_method;

  static
  {
    Class[] params = new Class[2];

    try {
      params[0] = Annotation.class;
      params[1] = OutputStream.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
      print_as_comment_method = params[0].getMethod("printAsComment", params);
      print_as_pragma_method = params[0].getMethod("printAsPragma", params);
      print_raw_method = params[0].getMethod("printRaw", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  private HashMap map;
  private String text;

/*
	static final private HashSet Annotation_type1
*/

  /**
   * Creates an empty annotation.
   */
  public Annotation()
  {
    object_print_method = class_print_method;

    map = new HashMap(4);
		parent = null;
    children = null;
    text = null;
  }

  /**
   * Creates a text annotation.
   *
   * @param text The text to be used as a comment or pragma.
   */
  public Annotation(String text)
  {
    object_print_method = class_print_method;

    map = new HashMap(1);
    children = null;
    this.text = text;
  }

  public void add(Object key, Object value)
  {
    map.put(key, value);
  }

  /**
   * Prints an annotation to a stream.
   *
   * @param note The annotation to print.
   * @param stream The stream on which to print the annotation.
   */
  public static void defaultPrint(Annotation note, OutputStream stream)
  {
    printAsComment(note, stream);
  }

  public boolean equals(Object o)
  {
    try {
      Annotation a = (Annotation)o;
      return toString().equals(a.toString());
    } catch (ClassCastException e) {
      return false;
    }
  }

  public List getDeclaredSymbols()
  {
    return new LinkedList();
  }

  /**
   * Provides access to the annotation map.
   */
  public Map getMap()
  {
    return map;
  }

  public String getText()
  {
    return text;
  }

  public int hashCode()
  {
    return toString().hashCode();
  }

  /**
   * Prints an annotation as a multi-line comment.
   *
   * @param note The annotation to print.
   * @param stream The stream on which to print the annotation.
   */
  public static void printAsComment(Annotation note, OutputStream stream)
  {
    PrintStream p = new PrintStream(stream);

    p.println("/*");
    if (note.text != null)
      p.println(note.text);
    else
      p.println(note.map.toString());
    p.print("*/");
  }


	public String toString()
	{
		StringBuilder str = new StringBuilder(80);

		if (object_print_method.getName().equals("printRaw"))
		{
			if ( text != null )
				str.append(text+"\n");
			else
				str.append(map.toString()+"\n");
		}
		else if ( object_print_method.getName().equals("printAsComment") ||
		object_print_method.getName().equals("defaultPrint") )
		{
			str.append("/*\n");
			if ( text != null )
				str.append(text+"\n");
			else
				str.append(map.toString()+"\n");
			str.append("*/");
		}
		else if (object_print_method.getName().equals("printAsPragma"))
		{
    	str.append("#pragma ");
	    if (text != null)
			{
				if (text.equals("cetus"))
					str.append("cetus ");
				else if (text.equals("openmp"))
					str.append("omp ");
				else
				{
					str.append(text+" ");
//					System.out.println("Unknown pragma Annotation");
//					System.exit(0);
				}
			}
			convertPragmaToString(str);
		}
		else 
		{ 
			/* defaultPrint case : do not print Annotation */
		}
		return str.toString();
	}

	public static enum parallel_construct
	{
		/* type 1 */
/*
		parallel, for, sections, section, single, task, master, barrier,
		taskwait, atomic, ordered,
*/

		/* type 2 */
/*
		private, firstprivate, lastprivate, shared, copyprivate, copyin,
		threadprivate, flush,
*/
		
		/* type 3 */
/*
		critical, if, num_threads, schedule, collapse, default,
*/

		/* type 4 */
		reduction;

		int get_type()
		{
			switch (this)
			{
/*
				case parallel: case for: case sections: case section: case single: 
				case task: case master: case barrier: case taskwait: case atomic: 
				case ordered:
							return 1;

				case private: case firstprivate: case lastprivate: case shared: 
				case copyprivate: case copyin: case threadprivate: case flush:
							return 2;

				case critical: case if: case num_threads: case schedule: 
				case collapse: case default:
							return 3;
*/

				case reduction:
							return 4; 
			}
			return 0;
		}	
	}

	/**
		* text : "openmp", "cetus", and so on
		* map : <key, value> pair
		*   - key  : a String; "parallel", "private", "reduction", etc
		*   - value: currently, there are four types of value we support
		*     - type 1: (String key, String "true") pair
		*					ex) parallel, flush, etc
		*     - type 2: (String key, Set<String> value) pair
		*					ex) private(a, b, c), 
		*     - type 3: (String key, String value) pair
		*					ex) default(shared), critical(name), etc
		*     - type 4: (String "reduction", Map reduction_map) pair
		*					ex) reduction_map = (String operator, Set<Expression>) pair
		*						  #pragma cetus	reduction(+: x, y) reduction(*: sum)
		*/
	public void convertPragmaToString(StringBuilder str)
	{
		if (map.keySet().contains("parallel")) str.append("parallel ");
			 
		if (map.keySet().contains("for")) str.append("for ");

		for ( String ikey : (Set<String>)(map.keySet()) )
		{	
			if (ikey.compareTo("parallel")==0 || ikey.compareTo("for")==0 )
			{ 
				/* do nothing */ 
			}
			else if (ikey.compareTo("sections")==0 ||
					ikey.compareTo("section")==0 ||
					ikey.compareTo("single")==0 ||
					ikey.compareTo("task")==0 ||
					ikey.compareTo("master")==0 ||
					ikey.compareTo("barrier")==0 ||
					ikey.compareTo("taskwait")==0 ||
					ikey.compareTo("atomic")==0 ||
					ikey.compareTo("ordered")==0 )
			{
				/* Type 1: (String key, String "true") pairs */
				str.append(ikey + " ");
			}
			else if (ikey.compareTo("private")==0 ||
					ikey.compareTo("firstprivate")==0 ||
					ikey.compareTo("lastprivate")==0 ||
					ikey.compareTo("shared")==0 ||
					ikey.compareTo("copyprivate")==0 ||
					ikey.compareTo("copyin")==0 ||
					ikey.compareTo("threadprivate")==0 ||
					ikey.compareTo("flush")==0 )
			{
				/** Type 2: (String key, Set<String>value) pairs */
				str.append(ikey + "(");
				str.append(Tools.collectionToString((Set)map.get(ikey), ", "));
				str.append(") ");
			}
			else if (ikey.compareTo("critical")==0 ||
							 ikey.compareTo("if")==0 ||
							 ikey.compareTo("num_threads")==0 ||
							 ikey.compareTo("schedule")==0 ||
							 ikey.compareTo("collapse")==0 ||
							 ikey.compareTo("default")==0 )
			{	
				/** Type 3: (String key, String value) pairs */
				str.append(ikey + "(");
				str.append( (String)map.get(ikey) );
				str.append(") ");
			}
			else if (ikey.compareTo("reduction")==0)
			{	
				/** Type 4: ("reduction", reduction_map) pair */ 
				Map<String, Set<Expression>> reduction_map = 
											(Map<String, Set<Expression>>)map.get(ikey);
				for (String op : (Set<String>)(reduction_map.keySet()) )
				{
					str.append("reduction");
					str.append("(" + op + ": ");
					str.append(Tools.collectionToString((Set)(reduction_map.get(op)), ", "));
					str.append(") ");
				}
			}
			else {	
				System.out.println("[Annotation.java] undefined annotation");
				System.exit(0);
			}
		}
	}

	public void convertSetOfExprToString(StringBuilder str, Set<Expression> iset)
	{
		int cnt = 0;
		if (iset == null) return;
		for ( Expression ie : iset )
		{
			if ( (cnt++)!=0 ) str.append(", ");
			str.append(ie.toString());
		}
	}

	public void convertSetToString(StringBuilder str, Set<String> iset)
	{
		int cnt = 0;
		if (iset == null) return;
		for ( String is : iset )
		{
			if ( (cnt++)!=0 ) str.append(", ");
			str.append(is);
		}
	}

	public void convertListToString(StringBuilder str, List<Expression> ilist)
	{
		int cnt = 0;
		if (ilist == null) return;
		for ( Expression ie : ilist )
		{
			if ( (cnt++)!=0 ) str.append(", ");
			str.append(ie.toString());
		}
	}

  /**
   * Prints an annotation as a single-line pragma.
   *
   * @param note The annotation to print.
   * @param stream The stream on which to print the annotation.
   */
  public static void printAsPragma(Annotation note, OutputStream stream)
  {
    PrintStream p = new PrintStream(stream);

    p.print("#pragma ");
    if (note.text != null)
		{
      p.print(note.text);
    	if (note.text.equals("cetus"))
			{
				if (note.map.keySet().contains("private"))
				{
					TreeSet<Expression> set = (TreeSet<Expression>)note.map.get("private");
					if (set == null)
						p.println(" private set is null");
					else
						p.print(" private(");
					int count=0;
					for (Expression ie : set)
					{
						if ((count++)>0) p.print(", ");
						p.print(ie.toString());
					}
					p.println(")");
				}
			}
		}
		else {
      p.print(note.map.toString());
		}

  }

  /**
   * Prints an annotation's contents without enclosing them in comments.
   *
   * @param note The annotation to print.
   * @param stream The stream on which to print the annotation.
   */
  public static void printRaw(Annotation note, OutputStream stream)
  {
    PrintStream p = new PrintStream(stream);

    if (note.text != null)
    {
      p.print(note.text);
      p.print(" ");
    }
    else
      p.print(note.map.toString());
  }

  /**
   * Unsupported - this object has no children.
   */
  public void setChild(int index, Traversable t)
  {
    throw new UnsupportedOperationException();
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

  /**
   * Sets the text of the annotation.
   *
   * @param text The text to be used as a comment or pragma.
   */
  public void setText(String text)
  {
    this.text = text;
  }
}
