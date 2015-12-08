package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
 * Provides tools for querying information related to For Loop objects
 *
 */
public class LoopTools {

	/**
	 * Constructor
	 * @return
	 */
	public LoopTools()
	{

	}

	/*
	 * Use the following static functions only with **ForLoops** that are identified
	 * as **CANONICAL** using **isCanonical**
	 * getIncrement(loop)
	 * getIndexVariable(loop)
	 * getLowerBound(loop)
	 * getUpperBound(loop)
	 */
	/**
	 * Get loop increment value, if loop is canonical
	 */
	public static long getIncrement(Loop loop)
	{
		long loopInc = 0;

		if (loop instanceof ForLoop)
		{
			ForLoop for_loop = (ForLoop)loop;
			/* determine the step */
			Expression step_expr = for_loop.getStep();

			if (step_expr instanceof AssignmentExpression)
			{
				Expression rhs = NormalExpression.simplify(((AssignmentExpression)step_expr).getRHS());
				loopInc = (new Long(rhs.toString())).longValue();
			}
			else if (step_expr instanceof UnaryExpression)
			{
				UnaryExpression uexpr = (UnaryExpression)step_expr;

				UnaryOperator op = uexpr.getOperator();
				if (op == UnaryOperator.PRE_INCREMENT ||
						op == UnaryOperator.POST_INCREMENT)
					loopInc = 1;
				else
					loopInc = -1;
			}		 
		}
		/* Handle other loop types */
		else
		{
		}

		return loopInc;
	}

	/**
	 * Get loop index variable, if loop is canonical
	 */
	public static Expression getIndexVariable(Loop loop)
	{
		Expression indexVar = null;

		/* Handle for loops here */
		if (loop instanceof ForLoop)
		{
			/* determine the name of the index variable */
			ForLoop for_loop = (ForLoop)loop;
			Expression step_expr = for_loop.getStep();

			if (step_expr instanceof AssignmentExpression)
			{
				indexVar = (Expression)((AssignmentExpression)step_expr).getLHS().clone();
			}
			else if (step_expr instanceof UnaryExpression)
			{
				UnaryExpression uexpr = (UnaryExpression)step_expr;

				indexVar = (Expression)uexpr.getExpression().clone();
			}
		}
		/* Handle other loop types */
		else
		{
		}

		return indexVar;
	}

	/**
	 * Get integral value of lower bound of loop, if loop is canonical
	 */
	public static long getLowerBound(Loop loop)
	{
		long lb = 0;

		if(loop instanceof ForLoop)
		{
			ForLoop for_loop = (ForLoop)loop;
			/* determine lower bound of index variable for this loop */
			Statement stmt = for_loop.getInitialStatement();

			/* Once C99 For loop statements are handled, this will need to be supported */
			if (stmt instanceof DeclarationStatement)
			{
				Initializer init = ((VariableDeclaration)((DeclarationStatement)stmt).getDeclaration()).getDeclarator(0).getInitializer();
				lb = (new Long(init.toString())).longValue();
			}
			else // ExpressionStatement
			{
				Expression expr = ((AssignmentExpression)((ExpressionStatement)stmt).getExpression()).getRHS();
				Expression sexpr = NormalExpression.simplify(expr);
				lb = (new Long(sexpr.toString())).longValue();
			}
		}
		else
		{
		}

		return lb;
	}

	/**
	 * Check if loop lower bound is a constant literal value
	 */
	public static boolean isLowerBoundConstant(Loop loop)
	{
		Expression lb;
		if (loop instanceof ForLoop)
		{
			ForLoop for_loop = (ForLoop)loop;
			/* determine lower bound for index variable of this loop */
			Statement stmt = for_loop.getInitialStatement();
			if (stmt instanceof ExpressionStatement)
			{
				Expression rhs = ((AssignmentExpression)((ExpressionStatement)stmt).getExpression()).getRHS();
				
				lb = NormalExpression.simplify(rhs);
				if (lb instanceof IntegerLiteral) 
					return true;
			}
			else if (stmt instanceof DeclarationStatement) { /* Error */ }
		}
		return false;
	}

	/**
	 * Check if loop upper bound is a constant literal value
	 */
	public static boolean isUpperBoundConstant(Loop loop)
	{
		Expression ub;

		if (loop instanceof ForLoop)
		{
			ForLoop for_loop = (ForLoop)loop;
			/* determine upper bound for index variable of this loop */
			BinaryExpression cond_expr = (BinaryExpression)for_loop.getCondition();
			Expression rhs = cond_expr.getRHS();

			ub = NormalExpression.simplify(rhs);
			if (ub instanceof IntegerLiteral) 
				return true;
			else 
				return false;
		}
		else
		{
			return false;
		}
	}

	/**
	 * Get the upper bound of the loop; if symbolic, return MAX_INT 
	 */
	public static long getUpperBound(Loop loop)
	{
		long ub = 0;

		if (loop instanceof ForLoop)
		{
			ForLoop for_loop = (ForLoop)loop;
			/* determine upper bound for index variable of this loop 
			 * If the upperbound is a constant IntegerLiteral, return
			 * its value, else return Long.MAX_VALUE for the purposes of
			 * data dependence analysis 
			 */
			if (isUpperBoundConstant(loop))
			{
				BinaryExpression cond_expr = (BinaryExpression)for_loop.getCondition();
				BinaryOperator op = cond_expr.getOperator();

				Expression rhs = NormalExpression.simplify(cond_expr.getRHS());
				if (op == BinaryOperator.COMPARE_LT)
					ub = (new Long(rhs.toString())).longValue() - getIncrement(loop);
				else if (op == BinaryOperator.COMPARE_LE)
					ub = (new Long(rhs.toString())).longValue();
			}
			/* Modify the else condition once symbolic analysis is incorporated */
			else
				ub = (new Long(Long.MAX_VALUE));
		}
		else
		{
		}

		return ub;
	}

	/**
	 * Calculate the loop nest of this loop
	 */
	public static LinkedList<Loop> calculateLoopNest(Loop loop)
	{
		LinkedList<Loop> loopNest = new LinkedList<Loop>();

		loopNest.add(loop);	
		Traversable t = ((ForLoop)loop).getParent();

		while (t != null)
		{
			if (t instanceof ForLoop)
			{
				loopNest.addFirst((Loop)t);
			}

			t = t.getParent();
		}
		/*
		 * Search for outer loop and add its nest to the current
		 * loop's nest before adding the current loop to its own
		 * nest list
		 */

		return loopNest;
	}

	/**
	 * Get common enclosing loops for loop1 and loop2, needs loopInfoMap for the nest
	 */
	public static LinkedList<Loop> getCommonNest(Loop loop1, Loop loop2, HashMap<Loop, LoopInfo> loopInfoMap)
	{
		LinkedList<Loop> commonNest = new LinkedList<Loop>();
		LinkedList<Loop> nest1 = (loopInfoMap.get(loop1)).loopNest;
		LinkedList<Loop> nest2 = (loopInfoMap.get(loop2)).loopNest;

		for (Loop l1 : nest1)
		{
			if (nest2.contains(l1))
				commonNest.addLast(l1);
		}	

		return commonNest;	
	}

	/**
	 * Check if loop is canonical, FORTRAN DO Loop format
	 */
	/*
	 * Following checks are performed:
	 * - Initial assignment expression with constant lower bound
	 * - Simple conditional expression with constant or symbolic upper bound
	 * - index variable increment with positive stride
	 * - check if index variable is invariant within loop body
	 */
	public static boolean isCanonical (Loop loop)
	{
		if (loop instanceof ForLoop)
		{
			ForLoop forloop = (ForLoop) loop;
			Identifier index_variable = null;

			//check initial statement
			Statement initial_stmt = forloop.getInitialStatement();
			if (initial_stmt instanceof ExpressionStatement)
			{
				ExpressionStatement exp_stmt = (ExpressionStatement)initial_stmt;
				index_variable = isInitialAssignmentExpression(exp_stmt.getExpression());
				if (index_variable == null)
				{
					return false;
				}
			}
			else
			{
				return false;
			}
			//check loop condition
			if (checkLoopCondition(forloop.getCondition(), index_variable)==null)
			{
				return false;
			}	
			//check loop step
			if ((checkIncrementExpression(forloop.getStep(), index_variable)) == false)
			{
				return false;
			}
			
			//check index invariant
			if ((isIndexInvariant((Loop)forloop, index_variable)) == false)
			{
				return false;
			}
		}

		// in the future it should handle other loops
		return true;
	}
	
	static private Identifier isInitialAssignmentExpression (Expression exp)
	{
		if (exp instanceof AssignmentExpression)
		{
			AssignmentExpression assignment_exp = (AssignmentExpression)exp;
			AssignmentOperator op = assignment_exp.getOperator();
			if (op.equals(AssignmentOperator.NORMAL))
			{
				Expression lhs = NormalExpression.simplify(assignment_exp.getLHS());
				Expression rhs = NormalExpression.simplify(assignment_exp.getRHS());
				if ((lhs instanceof Identifier) && (rhs instanceof Literal))
				{
					return((Identifier) lhs);
				}
			}
		}
		return null;
	}
	
	static private Expression checkLoopCondition (Expression cond_exp, Identifier induction_variable)
	{
		Expression loopbound = null;
		if (cond_exp instanceof BinaryExpression)
		{
			BinaryExpression bin_condexp = (BinaryExpression)cond_exp;

			BinaryOperator operator = bin_condexp.getOperator();

			Expression lhs = NormalExpression.simplify(bin_condexp.getLHS());
			Expression rhs = NormalExpression.simplify(bin_condexp.getRHS());

			if ((operator.equals(BinaryOperator.COMPARE_LT)) ||
					(operator.equals(BinaryOperator.COMPARE_LE)))
			{
				if (lhs.equals((Expression)induction_variable))
				{
					if (rhs instanceof Literal || rhs instanceof Expression)
					{
						loopbound = (Expression) rhs.clone();
					}
				}
			}
		}
		return loopbound;
	}

	static private boolean checkIncrementExpression (Expression exp, Identifier id)
	{
		if (exp instanceof UnaryExpression)
		{
			UnaryExpression unary_exp = (UnaryExpression)exp;
			if ((unary_exp.getOperator().equals(UnaryOperator.POST_INCREMENT) ||
					unary_exp.getOperator().equals(UnaryOperator.PRE_INCREMENT)) 
					&& unary_exp.getExpression().equals((Expression)id))
			{
				return true;
			}
		}
		else if (exp instanceof AssignmentExpression)
		{
			AssignmentExpression assign_exp = (AssignmentExpression) exp;
			if (!assign_exp.getLHS().equals((Expression)id))
			{
				return false;
			}
			if (assign_exp.getOperator().equals(AssignmentOperator.NORMAL))
			{
				if (assign_exp.getRHS() instanceof BinaryExpression)
				{
					BinaryExpression bin_exp = (BinaryExpression)assign_exp.getRHS();
					if (bin_exp.getOperator().equals(BinaryOperator.ADD))
					{
						/* Simplify the LHS and RHS of the binary expression to 
						 * accurately state whether we have a canonical increment
						 * expression or not
						 */
						Expression rhs = NormalExpression.simplify(bin_exp.getRHS());
						Expression lhs = NormalExpression.simplify(bin_exp.getLHS());
						Identifier ident_part = null;
						Literal lit_part = null;
						if (rhs instanceof Literal && lhs instanceof Identifier)
						{
							ident_part = (Identifier)lhs;
							lit_part = (Literal)rhs;
						}
						else if (rhs instanceof Identifier && lhs instanceof Literal)
						{
							ident_part = (Identifier)rhs;
							lit_part = (Literal)lhs;
						}
						//if (ident_part.equals(id) && lit_part.toString().equals("1"))
						if (ident_part.equals(id))
						{
							return true;
						}
					}
				}
			}
			else if (assign_exp.getOperator().equals(AssignmentOperator.ADD))
			{
				Expression rhs = NormalExpression.simplify(assign_exp.getRHS());
				if (rhs instanceof Literal)
				{
					Literal inc = (Literal)rhs;
/*					if (inc.toString().equals("1"))
					{
						return true;
					}
*/					return true;
				}
			}
		}
		return false;
	}

	/**
	 * Checks if loop body contains a function call
	 */
	public static boolean containsFunctionCall (Loop loop)
	{
		if (loop instanceof ForLoop)
			return (Tools.containsClass((Statement)loop.getBody(), FunctionCall.class));
		else
			return true;
	}

	/**
	 * Check if this loop and inner loops form a perfect nest
	 */
	public static boolean isPerfectNest(Loop loop)
	{
		boolean pnest = false;
		List children;
		Object o = null;
		Statement stmt = loop.getBody();
		
		FlatIterator iter = new FlatIterator((Traversable)stmt);
		if (iter.hasNext())
		{
			boolean skip = false;
			do
			{
				o = (Statement)iter.next(Statement.class);
				
				if ((o instanceof AnnotationStatement) ||
						(Tools.containsClass((Statement)o, Annotation.class)))
					skip = true;
				else
					skip = false;
				
			} while ((skip) && (iter.hasNext()));
			
			if (o instanceof ForLoop)
			{
				pnest = (isPerfectNest((Loop)o));
				
				/* The ForLoop contains additional statements after the end
				 * of the first ForLoop. This is interpreted as
				 * a non-perfect nest for dependence testing
				 */
				if (iter.hasNext())
					pnest = false;
			}
			else if (o instanceof CompoundStatement)
			{
				children = ((Statement)o).getChildren();
				Statement s = (Statement)children.get(0);
				if (s instanceof ForLoop)
					pnest = (isPerfectNest((Loop)s));
				else
					pnest = false;
			}
			else if (containsLoop(loop))
			{
				Tools.println("Loop is not perfectly nested", 2);
				pnest = false;
			}
			else
			{
				Tools.println("Loop is perfectly nested", 2);
				pnest = true;
			}
			
		}
		return pnest;
	}
	
	/**
	 * Check if loop body contains another loop
	 */
	public static boolean containsLoop(Loop loop)
	{
		/* Test whether a ForLoop contains another ForLoop */
		if (loop instanceof ForLoop)
			return (Tools.containsClass((Statement)loop.getBody(), ForLoop.class));
		else
			return true;
	}
	
	/**
	 * Check if the index variable is defined within the loop body
	 */
	public static boolean isIndexInvariant(Loop loop, Identifier id)
	{
		/* Get def set for loop body */
		Set<Expression> def_set = Tools.getDefSet((Statement)loop.getBody());
		
		if (def_set.contains((Expression)id))
				return false;
		
		return true;
	}

	/**
	 * Checks whether this loop contains any inner loops 
	 */
	public static boolean isInnermostLoop(Loop loop)
	{
		if (containsLoop(loop))
			return false;
		else
			return true;
	}

	/**
	 * Checks whether this loop is enclosed by any outer loops 
	 */
	public static boolean isOutermostLoop(Loop loop)
	{
		if (loop instanceof ForLoop)
		{
			ForLoop for_loop = (ForLoop)loop;
			Traversable t = for_loop.getParent();
			
			while (t != null)
			{
				if (t instanceof ForLoop)
					return false;
				else
					t = t.getParent();
			}
		}
		return true;
	}

	/**
	 * Get the outermost loop for the nest that surrounds the input loop
	 */
	public static Loop getOutermostLoop(Loop loop)
	{
		Loop return_loop = null;
		if (loop instanceof ForLoop)
		{
			if (isOutermostLoop(loop))
				return_loop = loop;
			else
			{
				ForLoop for_loop = (ForLoop)loop;
				Traversable t = for_loop.getParent();
			
				while (t != null)
				{
					if (t instanceof ForLoop)
					{
						if (isOutermostLoop((Loop)t))
							return_loop = (Loop)t;
						else
							t = t.getParent();
					}
					else
					{
						t = t.getParent();
					}
				}
			}
		}
		return return_loop;
	}

	/**
	 * Check whether the loop contains control constructs that cause it to terminate
	 * before the loop condition is reached
	 */
	public static boolean containsControlFlowModifier(Loop loop)
	{
		boolean ret_val = false;
		
		if (loop instanceof ForLoop)
		{
			if ((Tools.containsClass((Statement)loop.getBody(), GotoStatement.class)))
				ret_val = true;
			if ((Tools.containsClass((Statement)loop.getBody(), BreakStatement.class)))
				ret_val = true;
			if (Tools.containsClass((Statement)loop.getBody(), Label.class))
				ret_val = true;
			if (Tools.containsClass((Statement)loop.getBody(), ReturnStatement.class))
				ret_val = true;
					
			return ret_val;
		}
		else
			return true;
	}
}
