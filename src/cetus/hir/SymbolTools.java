package cetus.hir;

import java.util.*;


public final class SymbolTools
{
  /**
   * Makes links from all {@link IDExpression} objects in the program to
   * their corresponding declarators while generating warnings if there is
   * any undeclared variables or functions. This method is called before any
   * Cetus passes by default and provides a short cut to the declaration point
   * , which enables faster access to the declaration when necessary.
   *
   * @param program the input program
   */
	public static void linkSymbol(Program program)
	{
		double timer = Tools.getTime();

    addEnumeration(program);

		DepthFirstIterator iter = new DepthFirstIterator(program);

		Procedure proc = null;

		while ( iter.hasNext() )
		{
			Object oo = iter.next();

			if ( oo instanceof Procedure )
			{
				proc = (Procedure)oo;
				continue;
			}
			else if ( !(oo instanceof Identifier) )
				continue;

			Identifier id = (Identifier)oo;
			String id_name = id.getName();

			// These cases are skipped intentionally.
			if ( id.getParent() instanceof Declarator || // it is a Symbol object.
			id_name.equals("") ||                        // void Symbol.
			id_name.equals("__PRETTY_FUNCTION__") ||     // gcc keyword.
			id_name.equals("__FUNCTION__") ||            // gcc keyword.
			id_name.startsWith("__builtin") )            // gcc keyword.
				continue;

			Declaration decls = searchDeclaration(id);

			if ( decls == null )
				continue;

			// Find the symbol table object containing this declaration.
			Traversable t = decls.getParent();
			while ( !( t == null ||
			t instanceof ProcedureDeclarator ||
			t instanceof SymbolTable ) )
				t = t.getParent();

			SymbolTable symtab = null;
			if ( t instanceof SymbolTable )
				symtab = (SymbolTable)t;
			else if ( t instanceof ProcedureDeclarator )
				symtab = proc; // procedure parameters are orhpan traversables.

			// Add link from IDExpression (IR,symtab) to Symbol.
			if ( decls instanceof Procedure )
			{
				id.setSymbol((Symbol)decls);
				if ( symtab != null )
					for ( Object key : symtab.getTable().keySet() )
						if ( id.equals(key) )
							((IDExpression)key).setSymbol((Symbol)decls);
				continue;
			}

			// Found declaration containing the symbol
			Declarator decl = null;
			if ( decls instanceof VariableDeclaration )
			{
				for ( Object o: decls.getChildren() )
				{
					if ( id.equals( ((Declarator)o).getSymbol() ) )
					//if ( ((Declarator)o).getSymbol().toString().equals(id.getName()))
					{
						decl = (Declarator)o;

						if ( decl instanceof NestedDeclarator )
							decl = ((NestedDeclarator)decl).getDeclarator();

						id.setSymbol((Symbol)decl);

						if ( symtab != null )
							for ( Object key : symtab.getTable().keySet() )
								if ( id.equals(key) )
									((IDExpression)key).setSymbol((Symbol)decl);
						break;
					}
				}
			}
		}

		timer = Tools.getTime(timer);

		Tools.printStatus("[LinkSymbol] "+String.format("%.2f seconds\n",timer), 1);
	}

	// Returns the type of an expression
  private static List getType(Traversable e)
  {
		if ( !(e instanceof Expression) )
			return null;

    if ( e instanceof Identifier )
		{
			Symbol var = ((Identifier)e).getSymbol();
      return (var==null)? null: var.getTypeSpecifiers();
		}

		else if ( e instanceof AccessExpression )
			return getType(((AccessExpression)e).getRHS());

    else if ( e instanceof ArrayAccess )
      return getType(((ArrayAccess)e).getArrayName());

		else if ( e instanceof ConditionalExpression )
		{
			ConditionalExpression ce = (ConditionalExpression)e;
			List specs = getType(ce.getTrueExpression());
			if ( specs == null || specs.get(0) == Specifier.VOID )
				return getType(ce.getFalseExpression());
			else
				return specs;
		}

    else if ( e instanceof FunctionCall )
      return getType(((FunctionCall)e).getName());

    else if ( e instanceof Typecast )
      return ((Typecast)e).getSpecifiers();

		else
		{
			for ( Object child : e.getChildren() )
			{
				List child_type = getType((Expression)child);
				if ( child_type != null )
					return child_type;
			}
			return null;
		}
  }

  // Add enumeration entries to appropriate symbol tables
  private static void addEnumeration(Program program)
  {
		DepthFirstIterator iter = new DepthFirstIterator(program);

		for ( ;; )
		{
			Enumeration en = null;
			try {
				en = (Enumeration)iter.next(Enumeration.class);
			} catch ( NoSuchElementException ex ) {
				break;
			}
			Traversable t = en.getParent();
			while ( !(t instanceof SymbolTable) )
				t = t.getParent();
			for ( Object child : en.getChildren() )
        ((SymbolTable)t).getTable().put(
        ((VariableDeclarator)child).getSymbol(), en);
		}
  }

  // Serach for declaration of the identifier
  private static Declaration searchDeclaration(Identifier id)
  {
    Declaration ret = null;
    Traversable parent = id.getParent();

    if ( parent instanceof AccessExpression )
		{
      if ( ((BinaryExpression)parent).getRHS() == id )
			{
        Expression lhs = ((BinaryExpression)parent).getLHS();
        ClassDeclaration cdecl =
          (ClassDeclaration)findUserDeclaration(id, getType(lhs));
        if ( cdecl != null )
          ret = cdecl.findSymbol(id);
      }
      else
        ret = id.findDeclaration();
    }
    else
      ret = id.findDeclaration();

		// Prints out warning for undeclared functions/symbols.
    if ( ret == null )
		{
      if ( parent instanceof FunctionCall &&
        ((FunctionCall)parent).getName() == id )
        System.err.print("[WARNING] Function without declaration ");
      else
        System.err.print("[WARNING] Undeclared symbol ");

			System.err.println(id+" from "+parent);
    }
    return ret;
  }

  // Find the body of user-defined class declaration
  private static Declaration findUserDeclaration(Identifier id, List specs)
  {
    if ( specs == null )
      return null;

    // Find the leading user specifier
    UserSpecifier uspec = null;
    for ( Object o: specs )
		{
      if ( o instanceof UserSpecifier )
			{
        uspec = (UserSpecifier)o;
        break;
      }
		}

    if ( uspec == null )
      return null;

    // Find declaration for the user specifier
    Traversable t = id;
    Declaration ret = null;
    while ( t != null && ret == null )
		{
      if ( t instanceof SymbolTable )
			{
        ret = ((SymbolTable)t).findSymbol(uspec.getIDExpression());
				if ( ret instanceof VariableDeclaration &&
					((VariableDeclaration)ret).getSpecifiers() == specs )
					ret = null;												// cure for benchmark gcc
													                  // typedef struct {...} *foo;
			}                                     // foo foo=malloc(...)
      t = t.getParent();
    }

    // Keep searching through the chain ( i.e. typedef, etc )
    if ( ret instanceof VariableDeclaration )
      return findUserDeclaration(id,((VariableDeclaration)ret).getSpecifiers());

		// Differentiate prototype and actual declaration
		if ( ret instanceof ClassDeclaration &&
			((ClassDeclaration)ret).getTable().size() < 1 )
		{
			IDExpression class_name = ((ClassDeclaration)ret).getName();
			t = ret.getParent();
			while ( !(t instanceof SymbolTable) && t != null )
				t = t.getParent();
			FlatIterator iter = new FlatIterator(t);
			for ( ;; )
			{
				ClassDeclaration class_decl = null;
				try {
					class_decl = (ClassDeclaration)iter.next(ClassDeclaration.class);
				} catch ( NoSuchElementException ex ) {
					break;
				}
				if ( class_decl.getName().equals(class_name) &&
					class_decl.getTable().size() > 0 )
				{
					ret = class_decl;
					break;
				}
			}
		}
    return ret;
  }

}
