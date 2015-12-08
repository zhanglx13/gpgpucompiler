package cetus.hir;

import java.io.*;
import java.util.*;

import cetus.exec.*;

/**
 * A collection of useful static methods that operate on the IR.
 * In general, code goes here if it is used by multiple classes
 * that have no common ancestor.  Java does not have multiple
 * inheritence, and it seems better to put the code here instead
 * of introducing extra classes to solve the problem.  Note that
 * interfaces cannot solve the problem because you cannot provide
 * an implementation of a method within an interface. 
 */
public abstract class Tools
{
  /**
   * Java doesn't allow a class to be both abstract and final,
   * so this private constructor prevents any derivations.
   */
  private Tools()
  {
  }

  /**
   * Adds symbols to a symbol table and checks for duplicates.
   *
   * @param table The symbol table to add the symbols to.
   * @param decl The declaration of the symbols.
   * @throws DuplicateSymbolException if there are conflicts with
   *   any existing symbols in the table.
   */
  public static void addSymbols(SymbolTable table, Declaration decl)
  {
    List names = decl.getDeclaredSymbols();
    Iterator iter = names.iterator();

    HashMap symbol_table = table.getTable();

    while (iter.hasNext())
    {
      IDExpression expr = (IDExpression)iter.next();

      if (!(decl instanceof Procedure) && symbol_table.containsKey(expr)){
//        throw new DuplicateSymbolException(expr.toString() + " is already in this table");
        //System.err.println("[WARNING] " + expr.toString() + " is already in this table");
    	}
    	else 
        symbol_table.put(expr, decl);
    }
  }

  /**
   * Searches the IR tree beginning at t for the Expression e.
   * 
   * @return true if t contains e and false otherwise
   */
  public static boolean containsExpression(Traversable t, Expression e)
  {
    return (t.toString().indexOf(e.toString()) != -1);
  }

  /**
   * Counts the number of times that the Expression e appears in
   * the IR tree t.
   *
   * @return the number of times e appears in t
   */
  public static int countExpressions(Traversable t, Expression e)
  {
    String t_string = t.toString();
    String e_string = e.toString();
    int e_string_length = e_string.length();

    int i, n = 0;

    while ((i = t_string.indexOf(e_string)) != -1)
    {
      ++n;

      t_string = t_string.substring(i + e_string_length);
    }

    return n;
  }

  /**
   * Finds the first instance of the Expression e in the IR tree t.
   *
   * @return an expression from t that matches e
   */
  public static Expression findExpression(Traversable t, Expression e)
  {
    DepthFirstIterator iter = new DepthFirstIterator(t);
    String e_string = e.toString();

    for (;;)
    {
      Expression t_e;

      try {
        t_e = (Expression)iter.next(Expression.class);
      } catch (NoSuchElementException nse) {
        break;
      }

      if (t_e.toString().compareTo(e_string) == 0)
        return t_e;
    } 

    return null;
  } 

  /**
   * Searches for a symbol by name in the table.  If the symbol is
   * not in the table, then parent tables will be searched breadth-first.
   *
   * @param table The initial table to search.
   * @param name The name of the symbol to locate.
   *
   * @return a Declaration if the symbol is found, or null if it is not found.
   *    The Declaration may contain multiple declarators, of which name will
   *    be one, unless the SingleDeclarator pass has been run on the program.
   */
  public static Declaration findSymbol(SymbolTable table, IDExpression name)
  {
    LinkedList tables_to_search = new LinkedList();
    tables_to_search.add(table);

    /* Treat tables_to_search as a queue of tables, adding parent
       tables to the end of the list if name isn't in the current table. */
    while (!tables_to_search.isEmpty())
    {
      SymbolTable st = (SymbolTable)tables_to_search.removeFirst();
      
    Declaration decl = (Declaration)st.getTable().get(name);

      if (decl != null)
        return decl;

      tables_to_search.addAll(st.getParentTables());
    }

    return null;
  }

  static List getParentTables(Traversable obj)
  {
    LinkedList list = new LinkedList();

    Traversable p = obj.getParent();
    while (p != null)
    {
      try {
        SymbolTable st = (SymbolTable)p;
        list.add(st);
        break;
      } catch (ClassCastException e) {
        p = p.getParent();
      }
    }

    return list;
  }

  /**
   * Returns a randomly-generated name that is not found in the table.
   *
   * @param table The table to search.
   *
   * @return a unique name.
   */
  public static Identifier getUnusedSymbol(SymbolTable table)
  {
    String name = null;
    Identifier ident = null;
    Random rand = new Random();

    do {
      name = "";
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);
      name += 'a' + (rand.nextInt() % 26);

      ident = new Identifier(name);
    } while (findSymbol(table, ident) == null);

    return ident;
  }

  /**
   * The standard indexOf methods on lists use the equals method
   * when searching for an object; sometimes we want to use ==
   * and this method provides that service.
   *
   * @param list The list to search.
   * @param obj The object sought.
   *
   * @return the index of the object or -1 if it cannot be found.
   */
  public static int indexByReference(List list, Object obj)
  {
    int index = 0;

    Iterator iter = list.iterator();
    while (iter.hasNext())
    {
      if (iter.next() == obj)
        return index;
      else
        index++;
    }

    return -1;
  }

  /**
   * Calls print on every element of the list; does not put any
   * spaces or other text between the elements.
   *
   * @param list The list to print, which must contain only Printable objects.
   * @param stream The stream on which to print.
   */
  public static void printList(List list, OutputStream stream)
  {
    if (list != null)
    {
      Iterator iter = list.iterator();
      while (iter.hasNext())
        ((Printable)iter.next()).print(stream);
    }
  }

  /**
   * Calls print on every element of the list and prints a newline
   * after each element.
   *
   * @param list The list to print, which must contain only Printable objects.
   * @param stream The stream on which to print.
   */
  public static void printlnList(List list, OutputStream stream)
  {
    if (list == null || list.size() == 0)
      return;

    PrintStream p = new PrintStream(stream);

    Iterator iter = list.iterator();
    while (iter.hasNext())
    {
      ((Printable)iter.next()).print(stream);
      p.println("");
    }
  }

  /**
   * Prints a Printable object to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param p A Printable object.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printlnStatus(Printable p, int min_verbosity)
  {
    if (Integer.valueOf(Driver.getOptionValue("verbosity")).intValue() >= min_verbosity)
    {
      p.print(System.err);
      System.err.println("");
    }
  }

  /**
   * Prints a string to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param message The message to print.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printlnStatus(String message, int min_verbosity)
  {
    if (Integer.valueOf(Driver.getOptionValue("verbosity")).intValue() >= min_verbosity)
      System.err.println(message);
  }

  public static void print(String message, int min_verbosity)
  {
    if (Integer.valueOf(Driver.getOptionValue("verbosity")).intValue() >= min_verbosity)
      System.out.print(message);
  }

  public static void println(String message, int min_verbosity)
  {
//    if (Integer.valueOf(Driver.getOptionValue("verbosity")).intValue() >= min_verbosity)
      System.out.println(message);
  }

  /**
   * Prints a Printable object to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param p A Printable object.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printStatus(Printable p, int min_verbosity)
  {
    if (Integer.valueOf(Driver.getOptionValue("verbosity")).intValue() >= min_verbosity)
      p.print(System.err);
  }

  /**
   * Prints a string to System.err if the
   * verbosity level is greater than min_verbosity.
   *
   * @param message The message to print.
   * @param min_verbosity An integer to compare with the value
   *   set by the -verbosity command-line flag.
   */
  public static void printStatus(String message, int min_verbosity)
  {
    if (Integer.valueOf(Driver.getOptionValue("verbosity")).intValue() >= min_verbosity)
      System.err.print(message);
  }

  /**
   * Calls print on every element of the list and prints a comma
   * and a space between elements.
   *
   * @param list The list to print, which must contain only Printable objects.
   * @param stream The stream on which to print.
   */
  public static void printListWithCommas(List list, OutputStream stream)
  {
    printListWithSeparator(list, stream, ", ");
  }

  /**
   * Calls print on every element of the list and prints a string between them.
   * If the list has a single element, no separator is printed.
   *
   * @param list The list to print, which must contain only Printable objects.
   * @param stream The stream on which to print.
   * @param separator A string to print between the objects.
   */
  public static void printListWithSeparator(List list, OutputStream stream, String separator)
  {
    if (list == null || list.size() == 0)
      return;

    PrintStream p = new PrintStream(stream);

    Iterator iter = list.iterator();
    if (iter.hasNext())
    {
      ((Printable)iter.next()).print(stream);
      while (iter.hasNext())
      {
        p.print(separator);
        ((Printable)iter.next()).print(stream);
      }
    }
  }


	/**
	 * Converts a collection of objects to a string with the given separator.
	 * By default, the element of the collections are sorted alphabetically.
	 *
	 * @param coll the collection to be converted.
	 * @param separator the separating string.
	 * @return the converted string.
	 */
	public static String collectionToString(Collection coll, String separator)
	{
		if ( coll == null || coll.size() == 0 )
			return "";

		// Sort the collection first.
		TreeSet<String> sorted = new TreeSet<String>();
		for ( Object o : coll )
		{
			if ( o instanceof Symbol )
				sorted.add(((Symbol)o).getSymbolName());
			else
				sorted.add(o.toString());
		}

		StringBuilder str = new StringBuilder(80);

		Iterator<String> iter = sorted.iterator();
		if ( iter.hasNext() )
		{
			str.append(iter.next());
			while ( iter.hasNext() )
			{
				str.append(separator);
				str.append(iter.next());
			}
		}

		return str.toString();
	}


	/**
	 * Converts a list of objects to a string with the given separator.
	 *
	 * @param list the list to be converted.
	 * @param separator the separating string.
	 * @return the converted string.
	 */
	public static String listToString(List list, String separator)
	{
		if ( list == null || list.size() == 0 )
			return "";

		StringBuilder str = new StringBuilder(80);

		Iterator iter = list.iterator();
		if ( iter.hasNext() )
		{
			str.append(iter.next().toString());
			while ( iter.hasNext() )
				str.append(separator+iter.next().toString());
		}

		return str.toString();
	}

  /**
   * Replaces all instances of expression <var>x</var> on the IR tree
   * beneath <var>t</var> by <i>clones of</i> expression <var>y</var>.
   * Skips the immediate right hand side of member access expressions.
   *
   * @param t The location at which to start the search.
   * @param x The expression to be replaced.
   * @param y The expression to substitute.
   */
  public static void replaceAll(Traversable t, Expression x, Expression y)
  {
    BreadthFirstIterator iter = new BreadthFirstIterator(t);

    for (;;)
    {
      Expression o = null;

      try {
        o = (Expression)iter.next(x.getClass());
      } catch (NoSuchElementException e) {
        break;
      }

      if (o.equals(x))
      {
        if (o.getParent() instanceof AccessExpression
            && ((AccessExpression)o.getParent()).getRHS() == o)
        {
          /* don't replace these */
        }
        else
        {
          if (o.getParent() == null)
            System.err.println("[ERROR] this " + o.toString() + " should be on the tree");

          Expression copy = (Expression)y.clone();
          o.swapWith(copy);

          if (copy.getParent() == null)
            System.err.println("[ERROR] " + y.toString() + " didn't get put on tree properly");
        }
      }
    }
  }

  /**
   * Removes the symbols declared by the declaration from the symbol
   * table.
   *
   * @param table The table from which to remove the symbols.
   * @param decl The declaration of the symbols.
   */
  public static void removeSymbols(SymbolTable table, Declaration decl)
  {
    List names = decl.getDeclaredSymbols();
    Iterator iter = names.iterator();

    HashMap symbol_table = table.getTable();

    while (iter.hasNext())
    {
      IDExpression symbol = (IDExpression)iter.next();

      if (symbol_table.remove(symbol) == null)
      {
//        System.err.println("Tools.removeSymbols could not remove entry for " + symbol.toString());
//        System.err.println("table contains only " + symbol_table.toString());
      }
    }
  }

  /**
   * Verifies that every element of the list is of the same type.
   *
   * @param list The list to verify.
   * @param type The desired type for all elements.
   * @return true if all elements pass isInstance checks, false otherwise
   */
  public static boolean verifyHomogeneousList(List list, Class type)
  {
    Iterator iter = list.iterator();

    while (iter.hasNext())
    {
      if (!type.isInstance(iter.next()))
        return false;
    }

    return true;
  }

  /**
   * Returns a new identifier derived from the given identifier.
   *
   * @param id the identifier from which type and scope are derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Identifier id)
  {
    return getTemp(id, id.getName());
  }

  /**
   * Returns a new identifier derived from the given IR object and identifier.
   *
   * @param where the IR object from which scope is derived.
   * @param id the identifier from which type is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Traversable where, Identifier id)
  {
		return getTemp(where, id.getSymbol().getTypeSpecifiers(), id.getName());
  }

  /**
   * Returns a new identifier derived from the given identifier and name.
   *
   * @param id the identifier from which scope is derived.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Identifier id, String name)
  {
		return getTemp(id, id.getSymbol().getTypeSpecifiers(), name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type, and name.
   *
   * @param where the IR object from which scope is derived.
   * @param spec the type specifier.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Traversable where, Specifier spec, String name)
  {
    List specs = new ArrayList(1);
    specs.add(spec);
    return getTemp(where, specs, name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type list, and
   * name.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getTemp(Traversable where, List specs, String name)
  {
    return getArrayTemp(where, specs, (List)null, name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type list,
   * array specifier and name.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param aspec the array specifier.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getArrayTemp
  (Traversable where, List specs, ArraySpecifier aspec, String name)
  {
    List aspecs = new ArrayList(1);
    aspecs.add(aspec);
    return getArrayTemp(where, specs, aspecs, name);
  }

  /**
   * Returns a new identifier derived from the given IR object, type list,
   * array specifiers and name.
   *
   * @param where the IR object from which scope is derived.
   * @param specs the type specifiers.
   * @param aspecs the array specifier.
   * @param name the string from which name is derived.
   * @return the new identifier.
   */
  public static Identifier getArrayTemp
  (Traversable where, List specs, List aspecs, String name)
  {
    Traversable t = where;
    while ( !(t instanceof SymbolTable) )
      t = t.getParent();
    // Traverse to the parent of a loop statement
    if (t instanceof ForLoop || t instanceof DoLoop || t instanceof WhileLoop) {
      t = t.getParent();
      while ( !(t instanceof SymbolTable) )
        t = t.getParent();
    }
    SymbolTable st = (SymbolTable)t;
    
    String header = (name==null)? "_temp_": name+"_";
    Identifier ret = null;
    for ( int trailer=0; ret==null; ++trailer ) {
      Identifier newid = new Identifier(header+trailer);
      if ( findSymbol(st,newid) == null )
        ret = newid;
    }

    Declarator decl = null;
    if ( aspecs == null )
      decl = new VariableDeclarator(ret);
    else
      decl = new VariableDeclarator(ret, aspecs);
    Declaration decls = new VariableDeclaration(specs, decl);
    st.addDeclaration(decls);
		ret.setSymbol((Symbol)decl);

    return ret;
  }


	/**
	 * Returns the set of Symbol objects contained in the given SymbolTable
	 * object.
	 *
	 * @param st the symbol table being searched.
	 * @return the set of symbols.
	 */
	public static Set<Symbol> getSymbols(SymbolTable st)
	{
		Set ret = new HashSet<Symbol>();
		if ( st == null )
			return ret;
		for ( Object key : st.getTable().keySet() )
		{
			Symbol symbol = ((IDExpression)key).getSymbol();
			if ( symbol != null )
				ret.add(symbol);
		}
		return ret;
	}


	/**
	 * Returns the set of Symbol objects contained in the given SymbolTable
	 * object excluding Procedures.
	 *
	 * @param st the symbol table being searched.
	 * @return the set of symbols.
	 */
	public static Set<Symbol> getVariableSymbols(SymbolTable st)
	{
		Set ret = new HashSet<Symbol>();
		if ( st == null )
			return ret;
		for ( Object key : st.getTable().keySet() )
		{
			Symbol symbol = ((IDExpression)key).getSymbol();
			if ( !( symbol == null ||
			symbol instanceof Procedure ||
			symbol instanceof ProcedureDeclarator) )
				ret.add(symbol);
		}
		return ret;
	}

	/**
		* Returns the set of Symbol objects that are global variables 
		* of the File scope 
		*/
	public static Set<Symbol> getGlobalSymbols(Traversable t)
	{
		while (true)
		{
			if (t instanceof TranslationUnit) break;
			t = t.getParent(); 
		}
		TranslationUnit t_unit = (TranslationUnit)t;
		return Tools.getVariableSymbols(t_unit);
	}

	/**
		* Returns the Procedure to which the input traversable 
		* object belongs
		*/
	public static Procedure getCallerProcedure(Traversable t)
	{
		while (true)
		{
			if (t instanceof Procedure) break;
			t = t.getParent(); 
		}
		return (Procedure)t;	
	}

	/**
		* Returns the set of Symbol objects that are formal parameters of 
		* the given Procedure
		*/
	public static Set<Symbol> getParameterSymbols(Procedure proc)
	{
		HashSet<Symbol> parameters = new HashSet<Symbol>();
		for (Object o : proc.getParameters())
		{
			VariableDeclaration var_decl = (VariableDeclaration)o;
			List<IDExpression> id_expr_list = var_decl.getDeclaredSymbols();

			for (IDExpression id_expr : id_expr_list)
			{
				if (id_expr.getSymbol() != null)
				{
					parameters.add(id_expr.getSymbol());
				}
			}
		}

		return parameters;
	}

	public static Set<Symbol> getSideEffectSymbols(FunctionCall fc)
	{
		Set<Symbol> side_effect_set = new HashSet<Symbol>();
	
		// set of GlobalVariable Symbols that are accessed within a Procedure
		Procedure proc = fc.getProcedure();

		// we assume that there is no global variable access within a procedure
		// if a procedure body is not available for a compiler
		// example: system calls
		if (proc != null)
		{
			Set<Symbol> global_variables = new HashSet<Symbol>();
			Set<Symbol> accessed_symbols = getAccessedSymbols(proc.getBody());
			for (Symbol var : accessed_symbols)
			{
				if ( isGlobal(var, proc) )
				{
					global_variables.add(var);
				}
			}

			if ( !global_variables.isEmpty() )
			{
				side_effect_set.addAll(global_variables);
			}
		}
			
		// find the set of actual parameter Symbols of each function call
		Set<Symbol> parameters = getAccessedSymbols(fc);
		if ( !parameters.isEmpty() )
		{
			side_effect_set.addAll(parameters);	
		}

		return side_effect_set;
	}

	/**
	 * Returns the set of symbols accessed in the traversable object.
	 *
	 * @param t the traversable object.
	 * @return the set of symbols.
	 */
	public static Set<Symbol> getAccessedSymbols(Traversable t)
	{
		Set<Symbol> ret = new HashSet<Symbol>();

		if ( t == null )
			return ret;

		DepthFirstIterator iter = new DepthFirstIterator(t);

		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( !(o instanceof Identifier) )
				continue;
			Symbol symbol = ((Identifier)o).getSymbol();
			if ( symbol != null )
				ret.add(symbol);
		}

		return ret;
	}


	/**
	 * Returns a set of used expressions in the traversable object.
	 *
	 * @param t the traversable object.
	 * @return the set of used expressions.
	 */
	public static Set<Expression> getUseSet(Traversable t)
	{
		TreeSet<Expression> ret = new TreeSet<Expression>();

		DepthFirstIterator iter = new DepthFirstIterator(t);

		// Handle these expressions specially.
		iter.pruneOn(AccessExpression.class);
		iter.pruneOn(ArrayAccess.class);
		iter.pruneOn(AssignmentExpression.class);

		while ( iter.hasNext() )
		{
			Object o = iter.next();

			if ( o instanceof AccessExpression )
			{
				AccessExpression ae = (AccessExpression)o;
				DepthFirstIterator ae_iter = new DepthFirstIterator(ae);
				iter.pruneOn(ArrayAccess.class);

				// Catches array subscripts in the access expression.
				while ( ae_iter.hasNext() )
				{
					Object oo = ae_iter.next();
					if ( oo instanceof ArrayAccess )
					{
						ArrayAccess aa = (ArrayAccess)oo;
						Set aa_use = getUseSet(aa);
						aa_use.remove(aa);
						ret.addAll(aa_use);
					}
				}

				ret.add(ae);
			}
			else if ( o instanceof ArrayAccess )
			{
				ArrayAccess aa = (ArrayAccess)o;

				for ( int i=0; i < aa.getNumIndices(); ++i )
					ret.addAll(getUseSet(aa.getIndex(i)));

				ret.add(aa);
			}
			else if ( o instanceof AssignmentExpression )
			{
				AssignmentExpression ae = (AssignmentExpression)o;
				ret.addAll(getUseSet(ae.getRHS()));
				Set lhs_use = getUseSet(ae.getLHS());

				// Other cases should include the lhs in the used set. (+=,...)
				if ( ae.getOperator() == AssignmentOperator.NORMAL )
					lhs_use.remove(ae.getLHS());

				ret.addAll(lhs_use);
			}
			else if ( o instanceof Identifier )
			{
				Identifier id = (Identifier)o;

				if (id.getSymbol() instanceof Procedure ||
				id.getSymbol() instanceof ProcedureDeclarator)
					;
				else
					ret.add(id);
			}
		}

		return ret;
	}

	/**
	 * Returns a set of defined expressions in the traversable object.
	 * 
	 * @param t the traversable object.
	 * @return the set of defined expressions.
	 */
	public static Set<Expression> getDefSet(Traversable t)
	{
		Set<Expression> ret = new TreeSet<Expression>();

		if ( t == null )
			return ret;

		// Add increment/decrement operator in search list.
		Set unary_def = new HashSet();
		unary_def.add("--");
		unary_def.add("++");

		DepthFirstIterator iter = new DepthFirstIterator(t);

		while ( iter.hasNext() )
		{
			Object o = iter.next();

			// Expression being modified
			if ( o instanceof AssignmentExpression )
				ret.add(((AssignmentExpression)o).getLHS());

			else if ( o instanceof UnaryExpression )
			{
				UnaryExpression ue = (UnaryExpression)o;
				if ( unary_def.contains(ue.getOperator().toString()) )
					ret.add(ue.getExpression());
			}
		}

		return ret;
	}


	/**
	 * Returns a list of specifiers of the expression.
	 */
	public static LinkedList getVariableType(Expression e)
	{
		LinkedList ret = new LinkedList();
		if ( e instanceof Identifier )
		{
			Symbol var = ((Identifier)e).getSymbol();
			if ( var != null )
				ret.addAll(var.getTypeSpecifiers());
		}
		else if ( e instanceof ArrayAccess )
		{
			ArrayAccess aa = (ArrayAccess)e;
			ret = getVariableType(aa.getArrayName());
			for ( int i=0; i < aa.getNumIndices(); ++i )
				if ( ret.getLast() instanceof PointerSpecifier )
					ret.removeLast();
		}
		else if ( e instanceof AccessExpression )
		{
			Symbol var = ((AccessExpression)e).getSymbol();
			if ( var != null )
				ret.addAll(var.getTypeSpecifiers());
		}
		else if ( e instanceof UnaryExpression )
		{
			UnaryExpression ue = (UnaryExpression)e;
			if ( ue.getOperator() == UnaryOperator.DEREFERENCE )
			{
				ret = getVariableType(ue.getExpression());
				if ( ret.getLast() instanceof PointerSpecifier )
					ret.removeLast();
				else
					ret.clear();
			}
		}
		return ret;
	}

	/**
	 * Returns a list of specifiers of the given expression.
	 *
	 * @param e the given expression.
	 * @return the list of specifiers.
	 */
	public static List getExpressionType(Expression e)
	{
		if ( e instanceof Identifier )
		{
			Symbol var = ((Identifier)e).getSymbol();
			if ( var != null )
				return var.getTypeSpecifiers();
		}
		else if ( e instanceof ArrayAccess )
		{
			ArrayAccess aa = (ArrayAccess)e;
			List ret = getExpressionType(aa.getArrayName());
			if ( ret != null )
			{
				LinkedList ret0 = new LinkedList(ret);
				for ( int i=0; i < aa.getNumIndices(); ++i )
					if ( ret0.getLast() instanceof PointerSpecifier )
						ret0.removeLast();
				return ret0;
			}
			return ret;
		}
		else if ( e instanceof AccessExpression )
		{
			Symbol var = ((AccessExpression)e).getSymbol();
			if ( var != null )
				return var.getTypeSpecifiers();
		}
		else if ( e instanceof AssignmentExpression )
		{
			return getExpressionType(((AssignmentExpression)e).getLHS());
		}
		else if ( e instanceof CommaExpression )
		{
			ArrayList children = (ArrayList)e.getChildren();
			return getExpressionType((Expression)children.get(children.size()-1));
		}
		else if ( e instanceof ConditionalExpression )
		{
			return getExpressionType(((ConditionalExpression)e).getTrueExpression());
		}
		else if ( e instanceof FunctionCall )
		{
			Expression fc_name = ((FunctionCall)e).getName();
			if ( fc_name instanceof Identifier )
			{
				Symbol fc_var = ((Identifier)fc_name).getSymbol();
				if ( fc_var != null )
					return fc_var.getTypeSpecifiers();
			}
		}
		else if ( e instanceof IntegerLiteral )
		{
			return new LinkedList(Arrays.asList(Specifier.LONG));
		}
		else if ( e instanceof BooleanLiteral )
		{
			return new LinkedList(Arrays.asList(Specifier.BOOL));
		}
		else if ( e instanceof CharLiteral )
		{
			return new LinkedList(Arrays.asList(Specifier.CHAR));
		}
		else if ( e instanceof StringLiteral )
		{
			return new LinkedList(Arrays.asList(
				Specifier.CHAR,
				PointerSpecifier.UNQUALIFIED
			));
		}
		else if ( e instanceof FloatLiteral )
		{
			return new LinkedList(Arrays.asList(Specifier.DOUBLE));
		}
		else if ( e instanceof Typecast )
		{
			return ((Typecast)e).getSpecifiers();
		}
		else if ( e instanceof UnaryExpression )
		{
			UnaryExpression ue = (UnaryExpression)e;
			UnaryOperator op = ue.getOperator();
			List ret = getExpressionType(ue.getExpression());
			if ( ret != null )
			{
				LinkedList ret0 = new LinkedList(ret);
				if ( op == UnaryOperator.ADDRESS_OF )
					ret0.addLast(PointerSpecifier.UNQUALIFIED);
				else if ( op == UnaryOperator.DEREFERENCE )
					ret0.removeLast();
				return ret0;
			}
			return ret;
		}
		else if ( e instanceof BinaryExpression )
		{
			Set logical_op =
				new HashSet(Arrays.asList("==",">=",">","<=","<","!=","&&","||"));
			BinaryExpression be = (BinaryExpression)e;
			BinaryOperator op = be.getOperator();
			if ( logical_op.contains(op.toString()) )
				return new LinkedList(Arrays.asList(Specifier.LONG));
			else
				return getExpressionType(be.getLHS());
		}
		printlnStatus("[WARNING] Unknown expression type: "+e, 0);
		return null;
	}

	/**
	 * Returns the symbol of the expression if it represents an lvalue.
	 *
	 * @param e the input expression.
	 * @return the corresponding symbol object.
	 */
	/*
	 * The following symbol is returned for each expression types.
	 * Identifier         : its symbol.
	 * ArrayAccess        : base name's symbol.
	 * AccessExpression   : access symbol (list of symbols).
	 * Pointer Dereference: the first symbol found in the expression tree.
	 */
	public static Symbol getSymbolOf(Expression e)
	{
		if ( e instanceof Identifier )
			return ((Identifier)e).getSymbol();
		else if ( e instanceof ArrayAccess )
			return getSymbolOf( ((ArrayAccess)e).getArrayName() );
		else if ( e instanceof AccessExpression )
			return ((AccessExpression)e).getSymbol();
		else if ( e instanceof UnaryExpression )
		{
			UnaryExpression ue = (UnaryExpression)e;
			if ( ue.getOperator() == UnaryOperator.DEREFERENCE )
			{
				DepthFirstIterator iter = new DepthFirstIterator(ue.getExpression());
				while ( iter.hasNext() )
				{
					Object o = iter.next();
					if ( o instanceof Identifier )
						return ((Identifier)o).getSymbol();
				}
			}
		}
		return null;
	}

	/**
	 * Returns a set of defined symbols from the traversable object.
	 *
	 * @param t the traversable object.
	 * @return the set of defined symbols.
	 */
	public static Set<Symbol> getDefSymbol(Traversable t)
	{
		Set<Symbol> ret = new HashSet<Symbol>();

		for ( Expression e : getDefSet(t) )
			ret.add(getSymbolOf(e));

		return ret;
	}


	/**
	 * Returns a set of used symbols from the traversable object.
	 *
	 * @param t the traversable object.
	 * @return the set of used symbols.
	 */
	public static Set<Symbol> getUseSymbol(Traversable t)
	{
		Set<Symbol> ret = new HashSet<Symbol>();

		for ( Expression e : getUseSet(t) )
			ret.add(getSymbolOf(e));

		return ret;
	}


	/**
   * Returns true if the traversable contains the specified symbol.
   * More accurate but slower than containsExpression.
   *
   * @param t    the traversable object being searched.
   * @param e    the expression object being searched for.
   */
	public static boolean containsExpr(Traversable t, Expression e)
	{
		if ( t == null )
			return false;
		String e_str = e.toString();
		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
			if ( iter.next().toString().equals(e_str) )
				return true;
		return false;
	}

	/**
	 * Returns true if the traversable contains the specified symbol
	 * @param t			traversable object being searched
	 * @param var		symbol being searched for
	 */
	public static boolean containsSymbol(Traversable t, Symbol var)
	{
		if ( t == null )
			return false;

		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Identifier && ((Identifier)o).getSymbol() == var )
				return true;
		}

		return false;
	}

	/**
	 * Returns true if the traversable contains one of the symbols in the set.
	 * @param t			traversable object being searched
	 * @param vars	set of symbols being searched for
	 */ 
	public static boolean containsSymbols(Traversable t, Set<Symbol> vars)
	{
		for ( Symbol var : vars )
			if ( containsSymbol(t, var) )
				return true;
		return false;
	}

	/**
	 * Returns true if the traversable contains the specified type of object
	 * @param t			the traversable object being searched
	 * @param type 	the class being searched for
	 */
	public static boolean containsClass(Traversable t, Class type)
	{
		if ( t == null )
			return false;
		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
			if ( iter.next().getClass() == type )
				return true;
		return false;
	}

	/**
	 * Checks if the collection contains the specified type of object.
	 *
	 * @param c the collection being searched.
	 * @param type the class being searched for.
	 */
	public static boolean containsClass(Collection c, Class type)
	{
		if ( c == null )
			return false;
		for ( Object o : c )
			if ( o.getClass() == type )
				return true;
		return false;
	}

	/**
	 * Checks if the traversable object contains the specified type of binary
	 * operations.
	 *
	 * @param t The traversable object being searched
	 * @param op The binary operator being searched for
	 * @return True if there is such an operation, False otherwise
	 */
	public static boolean containsBinary(Traversable t, BinaryOperator op)
	{
		if ( t == null )
			return false;
		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof BinaryExpression &&
				((BinaryExpression)o).getOperator() == op )
				return true;
		}
		return false;
	}

	/**
	 * Checks if the traversable object contains the specified type of unary
	 * operations.
	 *
	 * @param t The traversable object being searched
	 * @param op The unary operator being searched for
	 * @return True if there is such an operation, False otherwise
	 */
	public static boolean containsUnary(Traversable t, UnaryOperator op)
	{
		if ( t == null )
			return false;
		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof UnaryExpression &&
				((UnaryExpression)o).getOperator() == op )
				return true;
		}
		return false;
	}


	/**
	 * Returns a list of unary expressions with the given unary operator.
	 *
	 * @param t the traversable object being searched.
	 * @param op the unary operator being searched for.
	 * @return the list of unary expressions.
	 */
	public static List<UnaryExpression> getUnaryExpression
	(Traversable t, UnaryOperator op)
	{
		List<UnaryExpression> ret = new ArrayList<UnaryExpression>();
		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof UnaryExpression &&
			((UnaryExpression)o).getOperator() == op )
				ret.add((UnaryExpression)o);
		}
		return ret;
	}

	/**
   * Returns true if the traversable contains any side effects that change the
   * program state.
   *
   * @param t  The traversable object being searched
   * @return true if there is such a case, false otherwise.
   */
	public static boolean containsSideEffect(Traversable t)
	{
		if ( t == null )
			return false;

		Set unary_ops = new HashSet(Arrays.asList("--","++"));
		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof AssignmentExpression ||
				o instanceof FunctionCall ||
				o instanceof VaArgExpression ||
				o instanceof UnaryExpression &&
				unary_ops.contains(((UnaryExpression)o).getOperator().toString()) )
				return true;
		}
		return false;
	}

	/**
   * Returns a set of FunctionCall statement within the traversable 
   * @param t  traversable object being searched
   */
	public static List<FunctionCall> getFunctionCalls(Traversable t)
	{
		List<FunctionCall> fc_list = new LinkedList<FunctionCall>();

		if ( t == null ) return null;

		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if (o instanceof FunctionCall)
			{
				fc_list.add( (FunctionCall)o );
			}
		}
		return fc_list;
	}

	/**
	 * Checks if the symbol is a global variable to the procedure containing the
	 * given traversable object.
	 *
	 * @param symbol The symbol object
	 * @param t The traversable object
	 * @return true if it is global, false otherwise
	 */
	public static boolean isGlobal(Symbol symbol, Traversable t)
	{
		while ( t != null && !(t instanceof Procedure) )
			t = t.getParent();

		if ( t == null )
			return true; // conservative decision if a bad thing happens.

		List parent_symtabs = getParentTables(t);

		for ( Object symtab : parent_symtabs )
		{
			Set symbols = getSymbols((SymbolTable)symtab);
			if ( symbols.contains(symbol) )
				return true;
		}

		return false;
	}

	/**
	 * Checks if the symbol is a scalar variable.
	 *
	 * @param symbol The symbol
	 * @return true if it is a scalar variable, false otherwise
	 */
	public static boolean isScalar(Symbol symbol)
	{
		if ( symbol == null )
			return false;

		List specs = symbol.getArraySpecifiers();

		return ( specs == null || specs.isEmpty() );
	}

	/**
	 * Checks if the symbol is an array variable.
	 *
	 * @param symbol The symbol
	 * @return true if it is an array variable, false otherwise
	 */
	public static boolean isArray(Symbol symbol)
	{
		if ( symbol == null )
			return false;

		List specs = symbol.getArraySpecifiers();

		return ( specs != null && !specs.isEmpty() );
	}

	/**
	 * Checks if the symbol is a pointer type variable.
	 *
	 * @param symbol The symbol
	 * @return true if it is a pointer type variable, false otherwise
	 */
	public static boolean isPointer(Symbol symbol)
	{
		if ( symbol == null )
			return false;

		List specs = symbol.getTypeSpecifiers();

		if ( specs == null )
			return false;

		for ( Object o : specs )
			if ( o instanceof PointerSpecifier )
				return true;

		return false;
	}


	/**
	 * Checks if the symbol is a pointer type variable. The input expression
	 * should represent a variable. Otherwise it will return true.
	 *
	 * @param e the expression to be tested.
	 */
	public static boolean isPointer(Expression e)
	{
		List spec = getExpressionType(e);
		if ( spec == null || spec.isEmpty() ||
			spec.get(spec.size()-1) instanceof PointerSpecifier )
			return true;
		else
			return false;
	}

	/**
	 * Checks if the symbol is an interger type variable.
	 *
	 * @param symbol the symbol.
	 * @return true if it is an integer type variable, false otherwise.
	 */
	public static boolean isInteger(Symbol symbol)
	{
		if ( symbol == null )
			return false;

		Set include = new HashSet(Arrays.asList(
			Specifier.INT,
			Specifier.LONG,
			Specifier.SIGNED,
			Specifier.UNSIGNED
		));
		Set exclude = new HashSet(Arrays.asList(
			Specifier.CHAR,
	    PointerSpecifier.UNQUALIFIED,
	    PointerSpecifier.CONST,
	    PointerSpecifier.VOLATILE,
	    PointerSpecifier.CONST_VOLATILE
		));

		List specs = symbol.getTypeSpecifiers();

		if ( specs == null )
			return false;

		boolean ret = false;
		for ( Object o : specs )
		{
			if ( exclude.contains(o) )
				return false;
			ret |= include.contains(o);
		}
		return ret;
	}

	/**
	 * Returns a current system time in seconds since a system-wise reference
	 * time.
	 *
	 * @return the time in seconds.
	 */
	public static double getTime()
	{
		return (System.currentTimeMillis()/1000.0);
	}

	/**
	 * Returns the elapsed time in seconds since the given reference time.
	 *
	 * @param since the reference time
	 * @return the elapsed time in seconds
	 */
	public static double getTime(double since)
	{
		return (System.currentTimeMillis()/1000.0 - since);
	}

	/**
		* Returns a key set of annotations attached to a given statement. 
		* For example, 
		* #pragam cetus parallel
		* #pragam cetus for reduction(+:sum)
		* will return a set containing {parallel, for, reduction}
		*/
	public static Set getAnnotationSet(String name, Statement istmt )
	{
		HashSet<String> annot_set = new HashSet<String>();

		FlatIterator iter = new FlatIterator(istmt.getParent());
		ArrayList<Statement> stmt_list = iter.getList(Statement.class);

		for (Statement stmt : stmt_list)
		{
			if (stmt instanceof AnnotationStatement)
			{
				AnnotationStatement annot_stmt = (AnnotationStatement)stmt;
				if (annot_stmt.getStatement() == istmt)
				{
					Annotation annot = annot_stmt.getAnnotation();
					if (annot.getText().compareTo(name)==0)
					{
						for ( String str : (Collection<String>)(annot.getMap().keySet()) )
						{
							annot_set.add(str);
						}
					}
				}
			}
		}
		return annot_set;
	}

	public static boolean containsAnnotation(Annotation annot, String pname, String aname)
	{ 
		if ( (annot.getText().compareTo(pname)==0) && (annot.getMap().containsKey(aname)) )
			return true;
		else
			return false;
	}

	/**
		* Returns true if the Statement istmt has an Annotation aname in pragma pname 
		* attached to it 
		* @param istmt : input Statement 
		* @param pname : pragma name such as "omp", "cetus"
		* @param aname : annotation name such as "parallel", "shared", and "reduction"
		*/
	public static boolean containsAnnotation(Statement istmt, String pname, String aname)
	{ 
		Object ret_val = null;

		FlatIterator iter = new FlatIterator(istmt.getParent());
		ArrayList<Statement> stmt_list = iter.getList(Statement.class);

		for (Statement stmt : stmt_list)
		{
			if (stmt instanceof AnnotationStatement)
			{
				AnnotationStatement annot_stmt = (AnnotationStatement)stmt;
				if (annot_stmt.getStatement() == istmt)
				{
					Annotation annot = annot_stmt.getAnnotation();
					if (containsAnnotation(annot, pname, aname))
					{
						return true;
					}
				}
			}
		}
		return false;
	}

	public static Set<AnnotationStatement> getAnnotStatementSet(Statement istmt, String pname)
	{ 
		Set<AnnotationStatement> ret = new HashSet<AnnotationStatement>();

		DepthFirstIterator iter = new DepthFirstIterator(istmt.getParent());
		ArrayList<Statement> stmt_list = iter.getList(Statement.class);

		for (Statement stmt : stmt_list)
		{
			if (stmt instanceof AnnotationStatement)
			{
				AnnotationStatement annot_stmt = (AnnotationStatement)stmt;
				if (annot_stmt.getStatement() == istmt)
				{
					Annotation annot = annot_stmt.getAnnotation();
					if (annot.getText().equals(pname))
						ret.add((AnnotationStatement)stmt);	
				}
			}
		}
		return ret;
	}

	public static Object getAnnotation(Statement istmt, String pname, String aname)
	{ 
		Object ret_val = null;

		Set<AnnotationStatement> annot_stmts = getAnnotStatementSet(istmt, pname);

		for (AnnotationStatement annot_stmt : annot_stmts)
		{
			if (annot_stmt.getStatement() == istmt)
			{
				Annotation annot = annot_stmt.getAnnotation();
				if (annot.getMap().containsKey(aname))
				{
					ret_val = annot.getMap().get(aname);
					break;
				}
			}
		}
		return ret_val;
	}

}



