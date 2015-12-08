/* 
   OpenMP spec 3.0 (the last section in page 99)
  
   The restrictions to the reduction clause are as follows 
   * A list item that appears in a reduction clause of a worksharing construct 
     must be shared in the parallel regions to which any of the worksharing 
     regions arising from the worksharing construct bind.
   * A list item that appears in a reduction clause of the innermost enclosing
     worksharing or parallel construct may not be accessed in an explicit task.
   * Any number of reduction clauses can be specified on the directive, but a 
     list item can appear only once in the reduction clause(s) for that directive.
     
    C/C++ specific restrictions
    - A type of a list item that appears in a reduction clause must be valid 
      for the reduction operator.
    - Aggregate types (including arrays), pointer types and reference types 
      may not appear in a reduction clause. 
    - A list item that appears in a reduction clause must not be const-qualified.
    - The operator specified in a reduction clause cannot be overloaded with 
      respect to C/C++ the list items that appear in that clause
  */

/**
	* Reduction pass performs reduction recognition for each ForLoop.
	* It generates cetus annotation in the form of "#pragma cetus reduction(...)"
	* Currently, it supports scalar (sum += ...), ArrayAccess (A[i] += ...),  and 
	* AccessExpression (A->x += ...) for reduction variable.
	* If another pass wants to access reduction information for a given statement,
	* stmt, Tools.getAnnotation(stmt, "cetus", "reduction") will return an object 
	* that contains a reduction map.
	* A reduction map, rmap, is a HashMap and has the following form;
	*	HashMap<String, HashSet<Expression>> rmap;
	* (where String represents a reduction operator and HashSet<Expression> is a 
	* set of reduction variable for a reduction operator key)
	*/

package cetus.analysis; 

import java.util.*;
import cetus.hir.*;
import cetus.exec.*;

/**
 * Performs reduction variable analysis to detect and annotate statements like
 * x = x + i in loops. An Annotation is added right before loops that contain
 * reduction variables. 
 */
public class Reduction extends AnalysisPass
{
	static int debug_tab=0;

	private int debug_level;
	private AliasAnalysis alias;

  public Reduction(Program program)
  {
    super(program);
		debug_level = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
  }

  public String getPassName()
  {
    return new String("[Reduction]");
  }

  public void start()
  {
		alias = new AliasAnalysis(program);
		alias.start();
		
		/**
			* Iterate over the outer-most loops
			*/
    BreadthFirstIterator iter = new BreadthFirstIterator(program);
/*
    iter.pruneOn(ForLoop.class);
*/
		ArrayList<ForLoop> outermost_loops = iter.getList(ForLoop.class);

    for (ForLoop loop : outermost_loops)
    {
			getReduction(loop);
    }
  }


	public void getReduction(ForLoop loop)
	{ 
		debug_tab++;
		if (debug_level > 1) {
			System.out.println("------------ getReduction strt ------------\n");
		}

    FlatIterator stmt_iter = new FlatIterator(loop.getBody());
		ArrayList<Statement> stmt_list = stmt_iter.getList(Statement.class);

		if (debug_level > 1)
		{
			System.out.println("debug_tab=" + debug_tab);
			System.out.println(loop.getBody().toString());
		}

		/** 	
			*  rmap: <reduction operator, a set of reduction candidate variable> pair
			*/
		HashMap<String, HashSet<Expression>> rmap = new HashMap<String, HashSet<Expression>>();

		/** 	
			*  RefSet: Referenced variable set
			*/
		HashSet<Symbol> RefSet = new HashSet<Symbol>();
		HashSet<Symbol> stmtRefSet = new HashSet<Symbol>();
		Set<Symbol> side_effect_set;

		int stmt_cnt=0;
    for (Statement stmt : stmt_list)
    {

			if (debug_level > 1)
			{
				System.out.println("stmt " + (++stmt_cnt) + "\n" + stmt.toString());
			}

			stmtRefSet.clear();
			stmtRefSet.addAll(Tools.getUseSymbol(stmt));
			stmtRefSet.addAll(Tools.getDefSymbol(stmt));
/*

 	    if (stmt instanceof ForLoop)
 	    {
 	      getReduction((ForLoop)stmt); // recursively check for inner loops
 	    }
 	    else */ if (stmt instanceof ExpressionStatement)
 	    {

				/* check if this ExpressionStatement contains a FunctionCall */
				List<FunctionCall> fc_list = Tools.getFunctionCalls((Traversable)stmt);
				if (fc_list != null)
				{
					for (FunctionCall fc : fc_list)
					{
						side_effect_set = Tools.getSideEffectSymbols(fc);
						if ( !side_effect_set.isEmpty() )
						{
							stmtRefSet.addAll(side_effect_set);
						}
					}
				}

				String reduce_op = checkReductionStatement(stmt);
				if ( reduce_op != null ) 
				{
					Expression top_expr = ((ExpressionStatement)stmt).getExpression();
					AssignmentExpression assignment = (AssignmentExpression)top_expr;
					Expression lhse = assignment.getLHS();
					Symbol lhs_symbol = Tools.getSymbolOf(lhse);
					if (lhs_symbol != null)
					{
						/* If there is no alias, then this lhse is a reduction variable */
						add_to_rmap(rmap, reduce_op, lhse);

						/* remove the found reduction variable from the current stmtRefSet */
						stmtRefSet.remove(lhs_symbol);

						if (debug_level > 1) {
							System.out.println("candidate = ("+reduce_op+":"+lhse.toString()+")");
						}
					}
					else
					{
						System.out.println("[Reduction] Error: unknown type"); 
						System.exit(0);
					}
				}
			}

			RefSet.addAll(stmtRefSet);

			if (debug_level > 2) 
			{
				displaySet("RefSet after stmt " + stmt_cnt, RefSet);
			}
    }


		/**
			* if the lhse of the reduction candidate statement is not in the RefSet, 
			* lhse is a reduction variable
			*/

		HashSet<String> tmp_set = new HashSet<String>();
		for ( String op : rmap.keySet() )
		{
			tmp_set.add(op);
		}

		for ( String op : tmp_set )
		{
			HashSet<Expression> reduction_set = rmap.get(op);
			HashSet<Expression> candidate_set = (HashSet<Expression>)reduction_set.clone();
			
			for (Expression candidate : candidate_set)
			{
				Symbol candidate_symbol = Tools.getSymbolOf(candidate);
				if (RefSet.contains(candidate_symbol))
				{
					reduction_set.remove(candidate);	
				}
				if ( alias.isAliased(null, candidate_symbol, RefSet) )
				{
					reduction_set.remove(candidate);	
				}
				if (candidate instanceof ArrayAccess)
				{
					// for (i=0; i<N; i++) { A[i] += expr; } : A[i] is not a reduction
					if ( is_array_indexed_with_loop_index((ArrayAccess)candidate, loop) )
					{
						reduction_set.remove(candidate);	
					}
				}
			}
			if (reduction_set.isEmpty())
			{
				rmap.remove(op);
			}
		}

		/**
			* Insert reduction Annotation to the current loop
			*/
		if (!rmap.isEmpty())
		{
			if (debug_level > 0)
			{
				print_reduction(rmap);
			}
			CompoundStatement parent_stmt = (CompoundStatement)(((Statement)loop).getParent());
			Annotation note = new Annotation("cetus");
			note.setPrintMethod(Annotation.print_as_pragma_method);
			note.add("reduction", rmap);
			AnnotationStatement annot_stmt = new AnnotationStatement(note);
			annot_stmt.attachStatement((Statement)loop);
			parent_stmt.addStatementBefore((Statement)loop, annot_stmt);
		}	

		if (debug_level > 1) {
			System.out.println("------------ getReduction done ------------\n");
		}
		debug_tab--;
	}

	private void add_to_rmap(HashMap<String, HashSet<Expression>> rmap, String reduce_op, Expression reduce_expr)
	{
		HashSet<Expression> reduce_set;
		if (rmap.keySet().contains(reduce_op))  
		{
			reduce_set = rmap.get(reduce_op);
			rmap.remove(reduce_op);
		}
		else
		{
			reduce_set = new HashSet<Expression>();
		}

		if (!reduce_set.contains(reduce_expr)) 
		{
			reduce_set.add(reduce_expr);
		}
		rmap.put(reduce_op, reduce_set);
	}
		
	private String checkReductionStatement(Statement stmt)
	{
		boolean isReduction = false;
		String reduction_op = null;		// reduction operator
		Expression top_expr = ((ExpressionStatement)stmt).getExpression();
		if (top_expr instanceof AssignmentExpression)
		{
			AssignmentExpression assignment = (AssignmentExpression)top_expr;

			AssignmentOperator assign_op = assignment.getOperator();
      Expression lhse = assignment.getLHS();
      Expression rhse = assignment.getRHS();
			Expression lhse_removed_rhse = null;

			if (debug_level > 1)
			{
				Symbol rhs_symbol;
				if ( (rhs_symbol = Tools.getSymbolOf(rhse)) != null )
				{
					System.out.println("lhs_symbol: " + rhs_symbol.getClass().getName());
				}
			}

			if (lhse instanceof IDExpression || lhse instanceof ArrayAccess || 
					lhse instanceof AccessExpression)
			{
				if (assign_op == AssignmentOperator.NORMAL) {
					// at this point either "lhse = expr;" or "lhse = lhse + expr;" is possible

					Expression simplified_rhse = NormalExpression.simplify(rhse);
					Expression lhse_in_rhse = Tools.findExpression(simplified_rhse, lhse);
					// if it is null, then it is not a reduction statement
					if (lhse_in_rhse == null)
					{
						return null;
					}
					Expression parent_expr = (Expression)(lhse_in_rhse.getParent());

					if (parent_expr instanceof BinaryExpression)
					{
						reduction_op = ((BinaryExpression)parent_expr).getOperator().toString();
					}
					else
						return null;

					if (reduction_op.equals("+")) 
					{
						lhse_removed_rhse = NormalExpression.subtract(rhse, lhse);
					}
					else if (reduction_op.equals("*")) 
					{
						lhse_removed_rhse = NormalExpression.divide(rhse, lhse);
					}
					else {
						/**
							* operators, such as {&, |, ^, &&, ||} are not supported
							*/
						return null;
					}
				}
				else if ( (assign_op == AssignmentOperator.ADD) ||
									(assign_op == AssignmentOperator.SUBTRACT) )
				{
					// case: lhse += expr; or lhse -= expr; 
					lhse_removed_rhse = NormalExpression.simplify(rhse);
					if (lhse_removed_rhse == null)
						System.out.println("[+= or -=] rhse_removed_rhse is null");
					reduction_op = new String("+");
				}
				else if ( assign_op == AssignmentOperator.MULTIPLY )
				{
					// case: lhse *= expr;
					lhse_removed_rhse = NormalExpression.simplify(rhse);
					if (lhse_removed_rhse == null)
						System.out.println("[*=] rhse_removed_rhse is null");
					reduction_op = new String("*");
				}
				else {
					return null;
				}

				if (debug_level > 1)
				{
					if (lhse_removed_rhse == null)
						System.out.println("[ERROR] rhse_removed_rhse is null");
					System.out.println("lhse_removed_rhse=" + lhse_removed_rhse.toString());
				}

				if (lhse instanceof Identifier)
				{
					Identifier id = (Identifier)lhse;

					if (!Tools.containsSymbol(lhse_removed_rhse, id.getSymbol()))
						isReduction = true;
				}
				else if (lhse instanceof ArrayAccess)
				{
					Expression base_array_name = ((ArrayAccess)lhse).getArrayName();
					if (base_array_name instanceof Identifier)
					{
						Identifier id = (Identifier)base_array_name;
						if (!Tools.containsSymbol(lhse_removed_rhse, id.getSymbol()))
							isReduction = true;
					}
				}
				else if (lhse instanceof AccessExpression)
				{
					Symbol lhs_symbol = Tools.getSymbolOf( lhse );
					if (!Tools.containsSymbol(lhse_removed_rhse, lhs_symbol))
						isReduction = true;
				}

				if (isReduction)
				{
					return reduction_op;
				}
			}
		}
		return null;
	}	

	private void print_reduction(HashMap<String, HashSet<Expression>> map)
	{
		if (!map.isEmpty())
		{
			int op_cnt=0;
			System.out.print("reduction = ");
			for (String op : map.keySet())
			{
				int cnt=0;
				if (op_cnt++ > 0) System.out.print(", ");
				System.out.print("(" + op + ":");
				for (Expression expr : map.get(op) )
				{
					if (cnt++ > 0) System.out.print(", ");
					System.out.print(expr.toString());
				}
				System.out.print(")");
			}
			System.out.println();
		}
		else
			System.out.println("reduction = {}");
	}

	/**
		* returns true if an array access index are all IntegerLiteral, eg, A[2][3].
		*/
	private boolean is_an_array_element_with_constant_index(ArrayAccess expr)
	{
		for (int i=0; i<expr.getNumIndices(); i++)
		{
			if ( (expr.getIndex(i) instanceof IntegerLiteral) == false )
				return false;
		}
		return true;
	}

	private boolean is_array_indexed_with_loop_index(ArrayAccess aae, Loop loop)
	{
		Expression loop_index_expr = LoopTools.getIndexVariable(loop);	
		Symbol loop_index_symbol = Tools.getSymbolOf(loop_index_expr);

		if (loop_index_symbol != null)
		{
			for (int i=0; i<aae.getNumIndices(); i++)
			{
				Expression array_index_expr = aae.getIndex(i);
				Symbol array_index_symbol = Tools.getSymbolOf(array_index_expr);
				if (array_index_symbol != null)
				{
					if (loop_index_symbol == array_index_symbol)
					{
						return true;
					}
				}
			}
		}
		return false;
	}

	public void displaySet(String name, Set iset)
	{
		int cnt = 0;
		if (iset == null) return;
		System.out.print(name + ":");
		for ( Object o : iset )
		{
			if ( (cnt++)==0 ) System.out.print("{");
			else System.out.print(", ");
			if (o instanceof Expression)
				System.out.print(o.toString());
			else if (o instanceof Symbol)
				System.out.print(((Symbol)o).getSymbolName());
			else 
			{
				if (o==null)
					System.out.println("null");
				else
					System.out.println("obj: " + o.getClass().getName());
			}
		}
		System.out.println("}");
	}
}
