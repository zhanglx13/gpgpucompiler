package cetus.transforms;

import java.util.*;

import cetus.analysis.*;
import cetus.hir.*;

/**
	* Extracts a region (a CompoundStatement) and convert in a subroutine;
	* assumes the SingleDeclarator pass has been run on the program.
	* A transformed subroutine is named after a subroutine that
	* contains the loop with a number. For example, a loop in
	* a subroutine foo is transformed into <var>foo_L0</var>
	* and the next loop into <var>foo_L1</var>, and so on.
	* Note: LoopsToSubroutines class converts all the loops in a given IR,
	*       i.e., every loop in a nested loop is recursively outlined.
	*				Whereas, this class converts only the given loop into a function
	*/
public class Outliner
{
  private static String pass_name = "[Outliner]";

	/**
		* @proc : Procedure that contains the loop to be transformed
		* @loop : a loop to be transformed into a function
		*/
  public static void extractRegion(Procedure proc, Statement region, String new_func_name)
  {
    SymbolTable global_table = (SymbolTable)proc.getParent();

    {
      Tools.println(pass_name + " creating new procedure " + new_func_name, 2);

      /* need to create 2 things: the new procedure where we are moving the 
				region to, and a function call to the new procedure which we will use
        to replace the region  */

			List<Specifier> new_proc_ret_type = new LinkedList<Specifier>();
			new_proc_ret_type.add(Specifier.GLOBAL);
      new_proc_ret_type.add(Specifier.VOID);

      Procedure new_proc = new Procedure(new_proc_ret_type,
        new ProcedureDeclarator(new Identifier(new_func_name), new LinkedList()),
        new CompoundStatement());

      FunctionCall call_to_new_proc = new FunctionCall(new Identifier(new_func_name));

      /* find the variables used inside the region and also the declarations 
				that appear inside the region */

      Set<Expression> use_set = (DataFlow.getUseSet((Traversable)region));
      Set def_set = DataFlow.mayDefine((Traversable)region);
      Set decl_set = getDeclSet((Traversable)region);

      Tools.println(pass_name + " region use set: " + use_set.toString(), 2);
      Tools.println(pass_name + " region def set: " + def_set.toString(), 2);
      Tools.println(pass_name + " region decl set: " + decl_set.toString(), 2);

      /* fill in the parameter list of the new procedure and
         the argument list of the new call */

			for (Expression expr : use_set)
			{

				/* Handle on IDExpressions, not interested in ArrayAccesses */
				if ( !IDExpression.class.isInstance(expr) ) { continue; }

				IDExpression id_expr = (IDExpression)expr;
        Declaration decl = id_expr.findDeclaration();

        /* Global variables in the use or def sets do not need passed
           into the new procedure because they are already accessible. */
        if (global_table.findSymbol(id_expr) == decl)
          continue;

        /* if this is not a structure member and
           if this is not a variable declared inside the region */
        if (decl != null && !decl_set.contains(decl))
        {
          VariableDeclaration cloned_decl = (VariableDeclaration)decl.clone();
          Expression cloned_expr = (Expression)id_expr.clone();

          /* Don't want static variables passed as static parameters because 
						static isn't allowed in parameter lists. The while loop is here 
						to remove all instances of static (although it would be weird to 
						have more than one, it is legal). */
          while (cloned_decl.getSpecifiers().remove(Specifier.STATIC));

          /* Remove any initializers.  We assume the SingleDeclarator pass
             was previously run so there is exactly one Declarator here. */
          cloned_decl.getDeclarator(0).setInitializer(null);

          /* Need to pass as pointers variables redefined by the region
             and struct variables.  A consequence is expr's in the arg list
             need to be &expr and uses of the variables need modified. */
          if (def_set.contains(cloned_expr)
              || (isStructure(cloned_decl) && !isPointer(cloned_decl) && !isPointer(cloned_decl.getDeclarator(0))))
          {
            /* FIXME: This should really go on the declarator, but right now 
							that doesn't seem to work and this does work. */
            cloned_decl.getSpecifiers().add(PointerSpecifier.UNQUALIFIED);
           
            /* Need to pass the address of the variable. */ 
            UnaryExpression address_of = new UnaryExpression(UnaryOperator.ADDRESS_OF, cloned_expr);
            address_of.setParens(false);
            cloned_expr = address_of;

            /* Inside the extracted region we must dereference the variables
               that we have turned into pointers.  Structures are also
               handled by this since (*s).x is equivalent to s->x */
            UnaryExpression deref = new UnaryExpression(UnaryOperator.DEREFERENCE, (Expression)id_expr.clone());
            Tools.replaceAll((Traversable)region, id_expr, deref);
          }          

          new_proc.addDeclaration(cloned_decl);
          call_to_new_proc.addArgument(cloned_expr);
        }
      }

      /* Put call_to_new_proc inside new_proc and then swap it with the region.  
				The call will end up where the region was and the region will end up in 
				the new procedure. */
			Statement stmt = new ExpressionStatement(call_to_new_proc);
      new_proc.getBody().addStatement(stmt);
      stmt.swapWith((Statement)region);

      /* put new_proc before the calling proc (avoids prototypes) */
      ((TranslationUnit)proc.getParent()).addDeclarationBefore(proc, new_proc);

    }
  }

  private static Set getDeclSet(Traversable root)
  {
    DepthFirstIterator iter = new DepthFirstIterator(root);
		HashSet<Declaration> decl_set = (HashSet<Declaration>)(iter.getSet(Declaration.class));

    return decl_set;
  }

  private static boolean isPointer(VariableDeclaration vdec)
  {
    Iterator iter = vdec.getSpecifiers().iterator();
    while (iter.hasNext())
    {
      if (iter.next() instanceof PointerSpecifier)
        return true;
    }

    return false;
  }

  private static boolean isPointer(Declarator decl)
  {
    Iterator iter = decl.getSpecifiers().iterator();
    while (iter.hasNext())
    {
      if (iter.next() instanceof PointerSpecifier)
        return true;
    }

    return false;
  }

  private static boolean isStructure(VariableDeclaration decl)
  {
    Iterator iter = decl.getSpecifiers().iterator();
    while (iter.hasNext())
    {
      Specifier spec = (Specifier)iter.next();
      if (spec instanceof UserSpecifier)
      {
        if (((UserSpecifier)spec).isStructure())
          return true;

        /* temp hack for f2c structure that is a typedef */
        if (spec.toString().equals("doublecomplex"))
          return true;
      }
    }

    return false;
  }

}
