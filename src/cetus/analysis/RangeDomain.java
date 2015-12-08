package cetus.analysis;

import java.util.*;
import cetus.hir.*;
import cetus.exec.*;

/**
 * RangeDomain provides symbolic environment for symbolic expression
 * comparisons. An object of this class keeps track of mapping from a symbol
 * to its corresponding expression that represents valid value range of the
 * symbol. It is guaranteed that, at each program point, this mapping is always
 * true. A range domain object is usually created after the range analysis but
 * an empty range domain is still useful as it can compare simple symbolic
 * values such as "e" and "e+1".
 */
public class RangeDomain implements Cloneable
{
	/*==================================================================
    Data fields
    ==================================================================*/

	// Set of symbolic value ranges.
	private HashMap<Symbol,Expression> ranges;

	// Cached results of symbolic comparisons.
	private HashMap<String,Relation> compared;

	// Current outstanding comparisons.
	private HashSet<String> comparing;

	// Debug tag
	private static String tag = "[RangeDomain]";

	/** Flag for range accuracy */
	private static int range_accuracy = 1;

	/** Global cache for wide-window caching */
	private static int compare_history = 1024;

	/** Global cache for expression comparison */
	private static ComparisonCache global_cache =
		new ComparisonCache(compare_history);

	// Debug level
	private static int debug =
		Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();


	/*==================================================================
    Constructors and data access methods
    ==================================================================*/

	/**
	 * Constructs an empty range domain.
	 */
	public RangeDomain()
	{
		ranges = new HashMap<Symbol,Expression>();
		//compared = new HashMap<String,Relation>();
		comparing = new HashSet<String>();
	}


	/**
	 * Copy constructor
	 * @param other			the original range domain
	 */
	public RangeDomain(RangeDomain other)
	{
		this();
		if ( other != null )
			for ( Symbol var : other.ranges.keySet() )
				setRange(var, (Expression)other.getRange(var).clone());
	}


	/**
	 * Returns a clone of the range domain.
	 *
	 * @return the cloned range domain.
	 */
	public Object clone()
	{
		RangeDomain o = new RangeDomain();
		for ( Symbol var : ranges.keySet() )
			o.setRange(var, (Expression)getRange(var).clone());

		return o;
	}


	/**
	 * Cleans up the fields of RangeDomain.
	 */
	public void clear()
	{
		ranges.clear();
		//compared.clear();
		comparing.clear();
	}


	/**
	 * Returns the number of value ranges in the map.
	 *
	 * @return	the number of value ranges.
	 */
	public int size()
	{
		return ranges.size();
	}


	/**
	 * Pairwise comparison between two RangeDomain objects.
	 *
	 * @param other	the RangeDomain object being compared to this object.
	 * @return			true if they are equal, false otherwise.
	 */
	public boolean equals(RangeDomain other)
	{
		if ( other == null || ranges.size() != other.ranges.size() )
			return false;

		for ( Symbol var : ranges.keySet() )
		{
			Expression range1 = getRange(var), range2 = other.getRange(var);
			if ( range2 == null || !range1.equals(range2) )
				return false;
		}

		return true;
	}


	/**
	 * Updates the value range for the specified variable.
	 *
	 * @param var		the variable whose value range is updated.
	 * @param value	the new value range of the variable.
	 */
	public void setRange(Symbol var, Expression value)
	{
		if ( isOmega(value) )
			ranges.remove(var);
		else
			ranges.put(var, value);
	}


	/**
	 * Returns the value range for the specified variable.
	 *
	 * @param var		the variable whose value range is asked for.
	 * @return			the value range for the variable.
	 */
	public Expression getRange(Symbol var)
	{
		return ranges.get(var);
	}


	/**
	 * Removes the value range for the specified variable.
	 *
	 * @param var		the variable whose value range is being removed.
	 */
	public void removeRange(Symbol var)
	{
		ranges.remove(var);
	}


	/**
	 * Returns the set of variables whose value ranges are present.
	 *
	 * @return			the set of variables.
	 */
	public Set getSymbols()
	{
		return ranges.keySet();
	}


	/**
   * Returns string for this range domain.
   *
   * @return string representation of this object.
   */
	public String toString()
	{
		String ret = "[";
		Map ordered = new TreeMap();

		for ( Symbol var : ranges.keySet() )
			ordered.put(var.getSymbolName(), getRange(var));

		int i = 0;
		for ( Object o : ordered.keySet() )
		{
			if ( i++ > 0 )
				ret += ", ";
			Expression range = (Expression)ordered.get(o);
			if ( range instanceof RangeExpression )
			{
				Expression lb = ((RangeExpression)range).getLB();
				Expression ub = ((RangeExpression)range).getUB();
				if ( lb instanceof InfExpression )
					ret += o+"<="+ub;
				else if ( ub instanceof InfExpression )
					ret += o+">="+lb;
				else
					ret += lb+"<="+o+"<="+ub;
			}
			else
				ret += o+"="+range;
		}
		return ret+"]";
	}


	/**
   * Converts this range domain to an equivalent logical expression.
   * e.g.) a=[-INF,b] ==> a<=b
   *
   * @return the equivalent logical expression.
   */
	public Expression toExpression()
	{
		Map<Identifier,Expression> ordered = new TreeMap<Identifier,Expression>();

		for ( Symbol var : ranges.keySet() )
			ordered.put(new Identifier(var.getSymbolName()), getRange(var));

		Expression ret = null;

		for ( Identifier var : ordered.keySet() )
		{
			Expression child = null;

			Expression range = (Expression)ordered.get(var);

			if ( range instanceof RangeExpression )
			{
				Expression lb = ((RangeExpression)range).getLB();
				Expression ub = ((RangeExpression)range).getUB();

				if ( lb instanceof InfExpression )
					child = new BinaryExpression(var, BinaryOperator.COMPARE_LE, ub);

				else if ( ub instanceof InfExpression )
					child = new BinaryExpression(var, BinaryOperator.COMPARE_GE, lb);

				else
					child = new BinaryExpression(
						new BinaryExpression(var, BinaryOperator.COMPARE_GE, lb),
						BinaryOperator.LOGICAL_AND,
						new BinaryExpression(var, BinaryOperator.COMPARE_LE, ub));
			}
			else
				child = new BinaryExpression(var, BinaryOperator.COMPARE_EQ, range);

			if ( ret == null )
				ret = child;
			else
				ret = new BinaryExpression(ret, BinaryOperator.LOGICAL_AND, child);
		}
		
		if ( range_accuracy > 1 )
			removeMinMax(ret);

		return ret;
	}




	/*==================================================================
    Core methods for comparison algorithm
    ==================================================================*/

/*
	public boolean compareGT(Expression e1, Expression e2)
	{
		Relation rel = compare(e1, e2);
		return rel.isGT();
	}

	public boolean compareGE(Expression e1, Expression e2)
	{
		Relation rel = compare(e1, e2);
		return rel.isGE();
	}

	public boolean compareLT(Expression e1, Expression e2)
	{
		Relation rel = compare(e1, e2);
		return rel.isLT();
	}

	public boolean compareLE(Expression e1, Expression e2)
	{
		Relation rel = compare(e1, e2);
		return rel.isLE();
	}

	public boolean compareEQ(Expression e1, Expression e2)
	{
		Relation rel = 
	}
*/

	/**
	 * Returns the relation between the two expressions under the constraints
	 * implied by the set of value ranges in the RangeDomain object. For a single
	 * call to this method, a new cache is created to speed up the comparison.
	 * @param e1		the first expression being compared
	 * @param e2		the second expression being compared
	 * @return			the {@link Relation} that stores the result of comparison
	 */
	public Relation compare(Expression e1, Expression e2)
	{
		Relation ret = null;

		String global_key = e1.toString()+e2.toString()+this.toString();

		if ( compare_history > 0 &&
			(ret=(Relation)global_cache.get(global_key)) != null )
		{
			//System.out.println("GCACHE hit");
			return ret;
		}
		//System.out.println("GCACHE miss");

		// Clean up caches before and after the comparisons
		//compared.clear();
		comparing.clear();

		// FIXME: When the cache is flushed?
		ret = compareExpressions(e1, e2);
		//compared.clear();
		//comparing.clear();

		// Cache the result globally.
		if ( compare_history > 0 )
			global_cache.put(global_key, ret);

		return ret;
	}


	/**
	 * Compares two expressions symbolically with the given range domain
	 * @param e1			first expression
	 * @param rd1			first range domain
	 * @param e2			second expression
	 * @param rd2			second range domain
	 * @return				{@link Relation} of the two expressions
	 */
	public static Relation compare
	(Expression e1, RangeDomain rd1, Expression e2, RangeDomain rd2)
	{
		Relation ret = rd1.compare(e1, e2);

		if ( !ret.isUnknown() && rd1 != rd2 )
			ret = Relation.AND(ret, rd2.compare(e1, e2));

		return ret;
	}


	/**
	 * Evaluates the given logical expression using the constraints.
	 *
	 * @param e the logical expression.
	 * @return the evaluation result; -1 for unknown, 0 for false, 1 for true.
	 */
	public int evaluateLogic(Expression e)
	{
		if ( e == null )
			return 1; // true for null expression.
		else if ( e instanceof BinaryExpression )
		{
			Set compare_op = new HashSet(Arrays.asList("==",">=",">","<=","<","!="));
			Set logical_op = new HashSet(Arrays.asList("||","&&"));
			BinaryExpression be = (BinaryExpression)e;
			String op = be.getOperator().toString();

			if ( compare_op.contains(op) )
			{
				Relation rel = compare(be.getLHS(), be.getRHS());
				if ( op.equals("==") && rel.isEQ() ||
				op.equals(">=") && rel.isGE() ||
				op.equals(">") && rel.isGT() ||
				op.equals("<=") && rel.isLE() ||
				op.equals("<") && rel.isLT() ||
				op.equals("!=") && rel.isNE() )
					return 1;
				else if ( op.equals("==") && rel.isNE() ||
				op.equals(">=") && rel.isLT() ||
				op.equals(">") && rel.isLE() ||
				op.equals("<=") && rel.isGT() ||
				op.equals("<") && rel.isGE() ||
				op.equals("!=") && rel.isEQ() )
					return 0;
				else
					return -1;
			}
			else if ( logical_op.contains(op) )
			{
				int lhs_eval = evaluateLogic(be.getLHS());
				int rhs_eval = evaluateLogic(be.getRHS());
				if ( op.equals("||") && (lhs_eval == 1 || rhs_eval == 1) ||
				op.equals("&&") && (lhs_eval == 1 && rhs_eval == 1) )
					return 1;
				else if ( op.equals("||") && (lhs_eval == 0 && rhs_eval == 0) ||
				op.equals("&&") && (lhs_eval == 0 || rhs_eval == 0) )
					return 0;
				else
					return -1;
			}
			else
				return -1;
		}
		else if ( e instanceof UnaryExpression )
		{
			UnaryExpression ue = (UnaryExpression)e;
			if ( ue.getOperator().toString().equals("!") )
			{
				int eval = evaluateLogic(ue.getExpression());
				if ( eval == 1 )
					return 0;
				else if ( eval == 0 )
					return 1;
				else
					return -1;
			}
			return -1;
		}
		else
			return -1;
	}


	// Wrapper for comparisons with expressions and integers
	private Relation compareExpressions(Expression e, int num)
	{
		return compareExpressions(e, new IntegerLiteral(num));
	}


	// Recursive method that compares the two expressions.
	private Relation compareExpressions(Expression e1, Expression e2)
	{
		Relation ret = null;

		// Take difference between two expressions
		Expression diff = diffInfExpressions(e1, e2);

		if ( diff == null )
			diff = diffRangeExpressions(e1, e2);

		if ( diff == null )
			diff = diffMinMaxExpressions(e1, e2);

		if ( diff == null )
			diff = subtract(e1, e2);

		if ( diff instanceof IntegerLiteral )
			return getRelation(diff);

		String signature = diff.toString();

		// Check if the diff matches an outstanding unsolved comparison.
		if ( comparing.contains(signature) )
			return new Relation();
		else
			comparing.add(signature);

		// Compute replacement orders
		List order = getReplaceOrder(diff);

		// Replace symbols with ranges in replacement-order
		Expression replaced = diff;
		Iterator iter = order.iterator();
		while ( iter.hasNext() && !isDecidable(replaced) )
		{
			Symbol var = (Symbol)iter.next();
			replaced = replaceSymbol(replaced, var, getRange(var));
		}

		// Replace the whole expression with omega
		if ( (ret=getRelation(replaced)).isUnknown() )
		{
			replaced = omega();
			// Reset to FFFF; this is critical for AND operation.
			ret = new Relation();
		}

		comparing.remove(signature);

		return ret;
	}


	// Decision criteria for stopping successive replacement; it stops right
	// after finding the sign of the expression since difference is always
	// passed to this method.
	private static boolean isDecidable(Expression e)
	{
		if ( e instanceof IntegerLiteral )
			return true;

		else if ( e instanceof RangeExpression )
		{
			RangeExpression re = (RangeExpression)e;
			Expression lb = re.getLB(), ub = re.getUB();

			return (
				(lb instanceof IntegerLiteral || lb instanceof InfExpression) &&
				(ub instanceof IntegerLiteral || ub instanceof InfExpression) ||
				lb instanceof IntegerLiteral && ((IntegerLiteral)lb).getValue() > 0 ||
				lb instanceof MinMaxExpression && ((MinMaxExpression)lb).isPosMax() ||
				ub instanceof IntegerLiteral && ((IntegerLiteral)ub).getValue() < 0 ||
				ub instanceof MinMaxExpression && ((MinMaxExpression)ub).isNegMin());
		}

		else
			return false;
	}


	// Returns equality/inequality of a symbolic comparison
	private static Relation getRelation(Expression e)
	{
		Relation ret = new Relation();
		
		// Integer literal
		if ( e instanceof IntegerLiteral )
		{
			long value = ((IntegerLiteral)e).getValue();
			ret.setLT(value<0);
			ret.setGT(value>0);
			ret.setEQ(value==0);
		}

		// Range expression
		else if ( e instanceof RangeExpression )
		{
			RangeExpression re = (RangeExpression)e;
			Expression lb = re.getLB(), ub = re.getUB();
			long lbval = Long.MIN_VALUE, ubval = Long.MAX_VALUE;

			if ( lb instanceof IntegerLiteral )
				lbval = ((IntegerLiteral)lb).getValue();
			else if ( lb instanceof MinMaxExpression &&
				((MinMaxExpression)lb).isPosMax() )
				lbval = 1;

			if ( ub instanceof IntegerLiteral )
				ubval = ((IntegerLiteral)ub).getValue();
			else if ( ub instanceof MinMaxExpression &&
				((MinMaxExpression)ub).isNegMin() )
				ubval = -1;

			if ( lbval > ubval );
			else if ( lbval < 0 )
			{
				ret.setLT(true);
				ret.setEQ(ubval>=0);
				ret.setGT(ubval>0);
			}
			else if ( lbval == 0 )
			{
				ret.setEQ(true);
				ret.setGT(ubval>0);
			}
			else
				ret.setGT(true);
		}

		// MIN/MAX expression
		else if ( e instanceof MinMaxExpression )
		{
			Long min = null, max = null;
			for ( Object o : e.getChildren() )
			{
				if ( !(o instanceof IntegerLiteral) )
					continue;

				long value = ((IntegerLiteral)o).getValue();

				if ( min == null )
				{
					min = new Long(value);
					max = new Long(value);
				}
				else
				{
					if ( value < min )
						min = value;
					if ( value > max )
						max = value;
				}

			}

			if ( min != null )
			{
				if ( ((MinMaxExpression)e).isMin() )
				{
					ret.setLT(true);
					ret.setEQ(min==0);
					ret.setGT(min>0);
				}
				else
				{
					ret.setGT(true);
					ret.setEQ(max==0);
					ret.setLT(max<0);
				}
			}
		}

		return ret;
	}


	// Wrapper to get the sign of an expression
	private int signOf(Expression e)
	{
		if ( e instanceof InfExpression )
			return ((InfExpression)e).sign();
	
		Relation rel = compareExpressions(e, 0);

		if ( rel.isGT() )
			return 1;

		else if ( rel.isLT() )
			return -1;

		else if ( rel.isEQ() )
			return 0;

		else
			return 999;
	}


	// Return the difference of two expressions that contain InfExpressions
	private Expression diffInfExpressions(Expression e1, Expression e2)
	{
		if ( e1 instanceof InfExpression )
		{
			if ( e2 instanceof InfExpression )
				return new IntegerLiteral(signOf(e1)-signOf(e2));
			else
				return new IntegerLiteral(signOf(e1));
		}

		else if ( e2 instanceof InfExpression )
			return new IntegerLiteral(-signOf(e2));

		else
			return null;
	}


	// Return the difference of two expressions that contain RangeExpressions
	private Expression diffRangeExpressions(Expression e1, Expression e2)
	{
		if ( e1 instanceof RangeExpression )
		{
			RangeExpression re1 = (RangeExpression)e1;
			Expression lb=null, ub=null;

			if ( e2 instanceof RangeExpression )
			{
				lb = subtract(re1.getLB(), ((RangeExpression)e2).getUB());
				ub = subtract(re1.getUB(), ((RangeExpression)e2).getLB());
			}
			else
			{
				lb = subtract(re1.getLB(), e2);
				ub = subtract(re1.getUB(), e2);
			}

			return new RangeExpression(lb, ub);
		}

		else if ( e2 instanceof RangeExpression )
		{
			Expression lb = subtract(e1, ((RangeExpression)e2).getUB());
			Expression ub = subtract(e1, ((RangeExpression)e2).getLB());
			return new RangeExpression(lb, ub);
		}

		else
			return null;
	}


	// Return the difference of two expressions that contain MinMax.
	private Expression diffMinMaxExpressions(Expression e1, Expression e2)
	{
		if ( !(e2 instanceof MinMaxExpression) )
			return null;

		if ( e1.equals(e2) )
			return new IntegerLiteral(0);

		return add(e1, ((MinMaxExpression)e2).negate());
	}

	/*==================================================================
    Supporting methods for expression manipulation
    ==================================================================*/

	/**
	 * Returns the forward-substituted value of a variable from the ranges.
	 * The return value is null if there is any dependence cycle in the range
	 * dependence graph.
	 * @param var			the variable whose value is asked for
	 * @return				the value range of the variable
	 */
	protected Expression getForwardSubstitution(Symbol var)
	{
		List replace_order = getReplaceOrder(getRange(var));

		// Check any duplicates and return null if so (cycles)
		Set vars = new HashSet();
		for ( Object key : replace_order )
		{
			if ( vars.contains(key) )
				return null;
			vars.add(key);
		}

		Expression ret = (Expression)getRange(var).clone();
		for ( Object key : replace_order )
			ret = replaceSymbol(ret, (Symbol)key, getRange((Symbol)key));

		return ret;
	}


	// Do forward substitution.
	protected void forwardSubstitute()
	{
		Set<Symbol> vars = new HashSet(ranges.keySet());
		for ( Symbol var : vars )
		{
			Expression substituted = getForwardSubstitution(var);
			if ( substituted != null )
				setRange(var, substituted);
		}
	}


	// Remove any ranges with a self cycle, e.g., a=[0,a]
	protected void removeRecurrence()
	{
		Set<Symbol> vars = new HashSet(ranges.keySet());
		for ( Symbol var : vars )
			if ( Tools.containsSymbol(getRange(var), var) )
				removeRange(var);
	}


	/**
	 * Removes the ranges for the speicfied variables in the map by replacing
	 * them with their value ranges.
	 * @param vars		the set of variables being removed
	 */
	public void removeSymbols(Set vars)
	{
		for ( Object o : vars )
		{
			Symbol var = (Symbol)o;
			removeSymbol(var);
			//replaceSymbol(var, getRange(var));
			//removeRange(var);
		}
	}


	/**
	 * Replaces all occurrences of the specified variable in the map with the
	 * given expression.
	 * @param var			the variable being replaced
	 * @param with		the expression being replaced with
	 */
	public void replaceSymbol(Symbol var, Expression with)
	{
		Set<Symbol> vars = new HashSet<Symbol>(ranges.keySet());

		for ( Symbol key : vars )
		{
			setRange(key,replaceSymbol((Expression)getRange(key).clone(), var, with));
			/*
			Expression replaced =
				replaceSymbol((Expression)getRange(key).clone(), var, with);
			*/
			/*
			if ( Tools.containsSymbol(replaced, key) )
				removeRange(key);
			else
				setRange(key, replaced);
			*/
			/*
			if ( !Tools.containsSymbol(replaced, key) )
				setRange(key, replaced);
			*/
		}
	}


	/**
   * Remove a variable in the range domain 
   */
	private void removeSymbol(Symbol var)
	{
		Expression with = getRange(var);

		removeRange(var);

		Set<Symbol> symbols = new HashSet<Symbol>(ranges.keySet());

		for ( Symbol symbol : symbols )
		{
			Expression replaced =
				replaceSymbol((Expression)getRange(symbol).clone(), var, with);

			Set<Symbol> kill = new HashSet<Symbol>(2);
			kill.add(var);
			kill.add(symbol);

			if ( Tools.containsSymbols(replaced, kill) )
				removeRange(symbol);
			else
				setRange(symbol, replaced);
		}
	}


	/**
	 * Replace all occurrences of the specified variable in the given expression
	 * with the new expression.
	 * @param e				the expression being modified
	 * @param var			the variable being replaced
	 * @param with		the new expression being replaced with
	 */
	public Expression replaceSymbol(Expression e, Symbol var, Expression with)
	{
		if ( with == null )
			return (Tools.containsSymbol(e, var))? null: e;

		// Direct simplification
		if ( !(with instanceof RangeExpression) )
		{
			DepthFirstIterator iter = new DepthFirstIterator(e);
			while ( iter.hasNext() )
			{
				Object o = iter.next();

				if ( !(o instanceof Identifier) )
					continue;

				Identifier id = (Identifier)o;
				
				if ( id.getSymbol() == var )
				{
					if ( e == id )
						return (Expression)with.clone();
					else
						id.swapWith((Expression)with.clone());
				}
			}
			return simplify(e);
		}

		// Replace var one-by-one: O(tree-depth*number_of_occurrences)
		Expression prev = null, expanded = e;

		while ( prev != expanded )
		{
			prev = expanded;
			expanded = expand(prev, var, with);
		}

		return expanded;
	}


/**
 * Eliminates all occurrences of the specified variables from the expression
 * by replacing them with their value ranges.
 */
	public Expression expandSymbols(Expression e, Set vars)
	{
		Expression ret = (Expression)e.clone();
		for ( Object var : vars )
		{
			if ( !(var instanceof Symbol) )
				return null;

			ret = expandSymbol(ret, (Symbol)var);
		}
		return ret;
	}

/**
 * Eliminates all occurrences of the specified variable from the expression
 * by replacing them with their value ranges.
 */
	public Expression expandSymbol(Expression e, Symbol var)
	{
		if ( e == null )
			return null;
		return replaceSymbol(e, var, getRange(var));
	}


	// Compute and return a replacement order after building Range Dependence
	// Graph.
	private List getReplaceOrder(Expression e)
	{
		List ret = new ArrayList();
		DFAGraph rdg = new DFAGraph();
		Set keyset = getDependentSymbols(e);
		DFANode root = new DFANode("scc-obj", e);
		
		if ( keyset.size() == 0 )
			return ret;

		for ( Object o: keyset )
			rdg.addEdge(root, new DFANode("scc-obj", o));

		for ( Object key: ranges.keySet() )
		{
			for ( Object o: getDependentSymbols(ranges.get(key)) )
			{
				DFANode from = rdg.getNode("scc-obj", key);
				DFANode to = rdg.getNode("scc-obj", o);
				if ( from == null )
					from = new DFANode("scc-obj", key);
				if ( to == null )
					to = new DFANode("scc-obj", o);
				rdg.addEdge(from, to);
			}
		}

		List scc = rdg.getSCC(root);

		for ( Object o: scc )
		{
			List tree = (List)o;

			if ( tree.size() == 1 )
				ret.add(0, ((DFANode)tree.get(0)).getData("scc-obj"));

			// Heurisitic method that repeats cycles twice.
			else
			{
				for ( Object key: tree ) ret.add(0, ((DFANode)key).getData("scc-obj"));
				for ( Object key: tree ) ret.add(0, ((DFANode)key).getData("scc-obj"));
			}
		}

		ret.remove(0); // remove the root
		return ret;
	}


	// Split the expression into set of key expressions that can be used as 
	// keys in the range domain
	private Set getDependentSymbols(Expression e)
	{
		HashSet ret = new LinkedHashSet();

		if ( e instanceof Identifier )
		{
			Symbol var = ((Identifier)e).getSymbol();
			if ( ranges.containsKey(var) )
				ret.add(var);
		}

		else if ( e instanceof BinaryExpression )
		{
			BinaryOperator op = ((BinaryExpression)e).getOperator();
			if ( op == BinaryOperator.ADD || op == BinaryOperator.DIVIDE ||
				op == BinaryOperator.MULTIPLY || op == BinaryOperator.SUBTRACT ||
				op == BinaryOperator.MODULUS )
			{
				ret.addAll(getDependentSymbols(((BinaryExpression)e).getLHS()));
				ret.addAll(getDependentSymbols(((BinaryExpression)e).getRHS()));
			}
		}

		else if ( e instanceof RangeExpression || e instanceof MinMaxExpression )
			for ( Object o: e.getChildren() )
				ret.addAll(getDependentSymbols((Expression)o));

		return ret;
	}


	// Expand the given expression after replacing the first occurrence of "var"
	// with "with". Here expand means RangeExpression is pulled up as much as
	// possible (usually up to the root of the expression tree).
	private Expression expand(Expression e, Symbol var, Expression with)
	{
		Identifier marker = null;

		DepthFirstIterator iter = new DepthFirstIterator(e);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( !(o instanceof Identifier) )
				continue;

			Identifier id = (Identifier)o;
			if ( id.getSymbol() == var )
			{
				marker = id;
				break;
			}
		}

		// No more expansion - guarantees termination
		if ( marker == null )
			return e;

		// Return a copy of the expression directly if the expressions is an
		// identifier
		if ( marker == e )
			return (Expression)with.clone();

		// Replace first
		Expression parent = (Expression)marker.getParent();
		marker.swapWith((Expression)with.clone());

		// Expand the replaced range up to the root of the expression tree
		while ( parent != e && parent != null )
		{
			Expression before = parent;
			parent = (Expression)parent.getParent();
			Expression expanded = expandOnce(before);
			before.swapWith(expanded);
		}

		// Final expansion at the top-level
		Expression ret = expandOnce(parent);

		return ret;
	}


	// Single expansion for the given expression
	private Expression expandOnce(Expression e)
	{
		Expression ret = omega();

		if ( e instanceof RangeExpression )
		{
			RangeExpression re = (RangeExpression)e;
			ret = expandRange(re.getLB(), re.getUB());
		}

		else if ( e instanceof BinaryExpression )
		{
			BinaryExpression be = (BinaryExpression)e;
			Expression l = be.getLHS(), r = be.getRHS();

			if (!(l instanceof RangeExpression)&&!(r instanceof RangeExpression))
				ret = (Expression)be.clone();

			if ( be.getOperator() == BinaryOperator.ADD )
				ret = expandADD(l, r);

			else if ( be.getOperator() == BinaryOperator.MULTIPLY )
				ret = expandMUL(l, r);

			else if ( be.getOperator() == BinaryOperator.DIVIDE )
				ret = expandDIV(l, r);

			else if ( be.getOperator() == BinaryOperator.MODULUS )
				ret = expandMOD(l, r);
		}

		else if ( e instanceof MinMaxExpression )
			ret = expandMinMax((MinMaxExpression)e);

		return simplify(ret);

	}


	// [e1:e2] => [e1.lb:e2.ub]
	private Expression expandRange(Expression e1, Expression e2)
	{
		Expression lb = (Expression) ((e1 instanceof RangeExpression)?
			((RangeExpression)e1).getLB().clone(): e1.clone());

		Expression ub = (Expression) ((e2 instanceof RangeExpression)?
			((RangeExpression)e2).getUB().clone(): e2.clone());

		return new RangeExpression(lb, ub);
	}


	// [e1:e2] => [e1.lb+e2.lb:e1.ub+e2.ub]
	private Expression expandADD(Expression e1, Expression e2)
	{
		RangeExpression re1 = toRange(e1), re2 = toRange(e2);
		Expression lb1 = re1.getLB(), ub1 = re1.getUB();
		Expression lb2 = re2.getLB(), ub2 = re2.getUB();
		Expression lb, ub;

		if ( lb1 instanceof InfExpression )
			lb = lb1;
		else if ( lb2 instanceof InfExpression )
			lb = lb2;
		else
			lb = add(lb1, lb2);

		if ( ub1 instanceof InfExpression )
			ub = ub1;
		else if ( ub2 instanceof InfExpression )
			ub = ub2;
		else
			ub = add(ub1, ub2);

		return new RangeExpression(lb, ub);

	}


	// [a:b]*c => [a*c:b*c]   if c>0,
	//            [b*c:a*c]   if c<0,
	//            [-inf:inf]  otherwise.
	private Expression expandMUL(Expression e1, Expression e2)
	{
		RangeExpression re;
		Expression e;

		// Identify RangeExpression
		if ( e1 instanceof RangeExpression )
		{
			if ( e2 instanceof RangeExpression )
				return omega();
			else
			{
				re = (RangeExpression)e1;
				e = e2;
			}
		}
		else if ( e2 instanceof RangeExpression )
		{
			re = (RangeExpression)e2;
			e = e1;
		}
		else
			return multiply(e1, e2);

		int e_sign = signOf(e);

		// Give up unknown result; e being InfExpression can progress further
		// if a>=0 in [a:b].
		if ( e instanceof InfExpression || e_sign == 999 )
			return omega();

		// Adjust lb ub position w.r.t. the sign of the multiplier
		Expression lb = re.getLB(), ub = re.getUB();
		if ( e_sign < 0 )
		{
			lb = re.getUB();
			ub = re.getLB();
		}

		// Lower bound
		if ( lb instanceof InfExpression )
			lb = new InfExpression(-1);
		else if ( lb instanceof MinMaxExpression )
		{
			List children = lb.getChildren();
			for ( int i=0; i<children.size(); ++i )
				lb.setChild(i, multiply(e, (Expression)children.get(i)));
			if ( e_sign < 0 )
			{
				MinMaxExpression mlb = (MinMaxExpression)lb;
				mlb.setMin(!mlb.isMin());
			}
		}
		else
			lb = multiply(e, lb);

		// Upper bound
		if ( ub instanceof InfExpression )
			ub = new InfExpression(1);
		else if ( ub instanceof MinMaxExpression )
		{
			List children = ub.getChildren();
			for ( int i=0; i<children.size(); ++i )
				ub.setChild(i, multiply(e, (Expression)children.get(i)));
			if ( e_sign < 0 )
			{
				MinMaxExpression mub = (MinMaxExpression)ub;
				mub.setMin(!mub.isMin());
			}
		}
		else
			ub = multiply(e, ub);

		return new RangeExpression(lb, ub);

	}


	private Expression expandDIV(Expression e1, Expression e2)
	{
		// [a:b]/c => [a/c:b/c]   if c>0,
		//            [b/c:a/c]   if c<0,
		//            [-INF:+INF] otherwise
		if ( e1 instanceof RangeExpression )
		{
			if ( e2 instanceof RangeExpression )
				return omega();

			int e2_sign = signOf(e2);

			if ( e2_sign == 999 || e2_sign == 0 )
				return omega();

			RangeExpression re1 = (RangeExpression)e1;
			Expression lb = re1.getLB(), ub = re1.getUB();
			if ( e2_sign < 0 )
			{
				lb = re1.getUB();
				ub = re1.getLB();
			}

			// Lower bound
			if ( lb instanceof InfExpression )
				lb = new InfExpression(-1);
			else if ( lb instanceof MinMaxExpression )
			{
				List children = lb.getChildren();
				for ( int i=0; i<children.size(); ++i )
					lb.setChild(i, divide((Expression)children.get(i), e2));
				if ( e2_sign < 0 )
				{
					MinMaxExpression mlb = (MinMaxExpression)lb;
					mlb.setMin(!mlb.isMin());
				}
			}
			else
				lb = divide(lb, e2);

			// Upper bound
			if ( ub instanceof InfExpression )
				ub = new InfExpression(1);
			else if ( ub instanceof MinMaxExpression )
			{
				List children = ub.getChildren();
				for ( int i=0; i<children.size(); ++i )
					ub.setChild(i, divide((Expression)children.get(i), e2));
				if ( e2_sign < 0 )
				{
					MinMaxExpression mub = (MinMaxExpression)ub;
					mub.setMin(!mub.isMin());
				}
			}
			else
				ub = divide(ub, e2);

			return new RangeExpression(lb, ub);
		}

		// c/[a:b] => [c/a:c/b]   if c<0 and (a>0||b<0),
		//            [c/b:c/a]   if c>0 and (a>0||b<0),
		//            [-INF:+INF] otherwise
		else if ( e2 instanceof RangeExpression )
		{
			int e1_sign = signOf(e1), e2_sign = signOf(e2);

			if ( e2_sign == 999 || e2_sign == 0 || e1_sign == 999 )
				return omega();
			if ( e1_sign == 0 )
				return new IntegerLiteral(0);

			RangeExpression re2 = (RangeExpression)e2;
			Expression lb = re2.getLB(), ub = re2.getUB();
			if ( e1_sign > 0 )
			{
				lb = re2.getUB();
				ub = re2.getLB();
			}

			// Lower bound
			if ( lb instanceof InfExpression )
				lb = new IntegerLiteral(0);
			else if ( lb instanceof MinMaxExpression )
			{
				List children = lb.getChildren();
				for ( int i=0; i<children.size(); ++i )
					lb.setChild(i, divide(e1, (Expression)children.get(i)));
				if ( e1_sign > 0 )
				{
					MinMaxExpression mlb = (MinMaxExpression)lb;
					mlb.setMin(!mlb.isMin());
				}
			}
			else
				lb = divide(e1, lb);
				
			// Upper bound
			if ( ub instanceof InfExpression )
				ub = new IntegerLiteral(0);
			else if ( ub instanceof MinMaxExpression )
			{
				List children = ub.getChildren();
				for ( int i=0; i<children.size(); ++i )
					ub.setChild(i, divide(e1, (Expression)children.get(i)));
				if ( e1_sign > 0 )
				{
					MinMaxExpression mub = (MinMaxExpression)ub;
					mub.setMin(!mub.isMin());
				}
			}
			else
				ub = divide(e1, ub);

			return new RangeExpression(lb, ub);
		}

		else
			return divide(e1, e2);
	}


	// Expansion for mod expressions
	private Expression expandMOD(Expression l, Expression r)
	{
		RangeExpression re = (RangeExpression)
			((l instanceof RangeExpression)? l: r);
		Expression other = (l instanceof RangeExpression)? r: l;

		Relation rel = compareExpressions(other, 0);
		Relation rellb = compareExpressions(re.getLB(), 0);
		Relation relub = compareExpressions(re.getUB(), 0);
		boolean positive_dividend = false, negative_dividend = false;
		Expression abs = null;
		Expression lb = new InfExpression(-1), ub = new InfExpression(1);

		if ( l instanceof RangeExpression )
		{
			if ( rel.isGT() )
				abs = other;
			else if ( rel.isLT() )
				abs = multiply(new IntegerLiteral(-1), other);

			positive_dividend = rellb.isGE();
			negative_dividend = relub.isLE();
		}

		else
		{
			if ( rellb.isGT() )
				abs = (Expression)re.getUB().clone();
			else if ( relub.isLT() )
				abs = multiply(new IntegerLiteral(-1), re.getLB());

			positive_dividend = rel.isGE();
			negative_dividend = rel.isLE();
		}

		// No division is possible
		if ( abs == null )
			return omega();

		// Range is defined by the divisor's maximum absolute value and
		// the sign of dividend can further narrow the range
		lb = multiply(new IntegerLiteral(-1), abs);
		lb = add(lb, new IntegerLiteral(1));
		ub = subtract(abs, new IntegerLiteral(1));

		if ( positive_dividend )
			lb = new IntegerLiteral(0);

		else if ( negative_dividend )
			ub = new IntegerLiteral(0);

		return new RangeExpression(lb, ub);
	}


	// Expand min/max expression
	private Expression expandMinMax(MinMaxExpression e)
	{
		MinMaxExpression lb = new MinMaxExpression(e.isMin());
		MinMaxExpression ub = new MinMaxExpression(e.isMin());

		for ( Object o: e.getChildren() )
		{
			for ( int i=0; i<2; ++i )
			{
				List temp = new ArrayList();
				MinMaxExpression bound = (i==0)? lb: ub;

				if ( o instanceof RangeExpression )
					temp.add(((RangeExpression)o).getChildren().get(i));
				else
					temp.add(o);

				for ( Object oo: temp )
				{
					if ( oo instanceof MinMaxExpression &&
						((MinMaxExpression)oo).isMin() == e.isMin() )
						for ( Object ooo: ((Expression)oo).getChildren() )
							bound.add((Expression)ooo);
					else
						bound.add((Expression)oo);
				}
			}
		}
		Expression ret = new RangeExpression(lb, ub);
		return simplify(ret);
	}



	/*====================================================================
    Methods for abstract operations; intersect, union, widen, and narrow
    ====================================================================*/
	/**
	 * Intersects two sets of value ranges using current range domain.
	 * @param other			the range domain intersected with
	 */
	public void intersectRanges(RangeDomain other)
	{
		// Copy explicit intersections first
		for ( Symbol var : other.ranges.keySet() )
			if ( getRange(var) == null )
				setRange(var, (Expression)other.getRange(var).clone());

		HashSet<Symbol> vars = new HashSet<Symbol>(ranges.keySet());
		for ( Symbol var : vars )
		{
			Expression result = intersectRanges(getRange(var), this,
				other.getRange(var), this);
			// Removing empty ranges trigger infeasible paths
			if ( isEmpty(result, this) )
			{
				ranges.clear();
				return;
			}
			if ( isOmega(result) )
				removeRange(var);
			else
				setRange(var, result);
		}
	}


	/**
	 * Merges two sets of value ranges using current range domain (union
	 * operation).
	 * @param other			the range domain merged with
	 */
	public void unionRanges(RangeDomain other)
	{
		HashSet<Symbol> vars = new HashSet<Symbol>(ranges.keySet());
		for ( Symbol var : vars )
		{
			Expression result = unionRanges(getRange(var), this,
				other.getRange(var), other);
			if ( isOmega(result) )
				removeRange(var);
			else
				setRange(var, result);
		}
	}


	/**
	 * Widens all value ranges of "other" range domain with this range domain.
	 * @param other			value ranges being widened
	 */
	public void widenRanges(RangeDomain other)
	{
		widenAffectedRanges(other, new HashSet<Symbol>(other.ranges.keySet()));
	}


	/**
	 * Widens subset of value ranges in "other" that contains the specified
	 * symbols either in keys or in value ranges.
	 * @param other			the range domain containing widening operands
	 * @param vars      set of symbols that trigger widening
	 */
	public void widenAffectedRanges(RangeDomain other, Set<Symbol> vars)
	{
		Set<Symbol> affected = new HashSet<Symbol>();
		for ( Symbol var_range : other.ranges.keySet() )
			for ( Symbol var_in : vars )
				if ( Tools.containsSymbol(getRange(var_range), var_in) )
					affected.add(var_range);
		affected.addAll(vars);

		for ( Symbol var : affected )
		{
			Expression result = widenRange(other.getRange(var), getRange(var), this);
			if ( isOmega(result) )
				removeRange(var);
			else
				setRange(var, result);
		}
	}


	/**
	 * Narrows all value ranges of "other" range domain with this range domain.
	 * @param other			value ranges being narrowed.
	 */
	public void narrowRanges(RangeDomain other)
	{
		//String dmsg = other+" (n) "+this;
		for ( Symbol var : other.ranges.keySet() )
		{
			Expression result = narrowRange(other.getRange(var), getRange(var), this);
			if ( isOmega(result) )
				removeRange(var);
			else
				setRange(var, result);
		}
		//printDebug(tag, 1, dmsg+" = "+this);
	}


	/**
	 * Computes the intersection of the two expressions with the given range
	 * domains.
	 * @param e1			first expression
	 * @param rd1			first range domain
	 * @param e2			second expression
	 * @param rd2			second range domain
	 * @return				intersection of the two expressions
	 */
	public static Expression intersectRanges
	(Expression e1, RangeDomain rd1, Expression e2, RangeDomain rd2)
	{
		// [lb:ub] = [a:b] ^ [c:d],    lb = a        if a>=c
		//                                = c        if a<c
		//                                = a        otherwise (range_accuracy=0)
		//                                = heuristics         (range_accuracy=1)
		//                                = max(a,c)           (range_accuracy=2)
		//                             ub = b        if b<=d
		//                                = d        if b>d
		//                                = b        otherwise (range_accuracy=0)
		//                                = heurisitcs         (range_accuracy=1) 
		//                                = min(b,d)           (range_accuracy=2)
		//
		// Check if e1/e2 is unknown range, empty range, or they are equal.
		if ( isOmega(e1) )
			return (isOmega(e2))? null: (Expression)e2.clone();
		else if ( isOmega(e2) )
			return (Expression)e1.clone();
		else if ( isEmpty(e1) )
			return (Expression)e1.clone();
		else if ( isEmpty(e2) )
			return (Expression)e2.clone();
		else if ( e1.compareTo(e2) == 0 )
			return (Expression)e1.clone();

		// Converts e1 & e2 to range expressions. re1 and re2 contain cloned copy
		// of e1 and e2 in their lb and ub.
		RangeExpression re1 = toRange(e1), re2 = toRange(e2);
		Expression lb1 = re1.getLB(), lb2 = re2.getLB(), lb = null;
		Expression ub1 = re1.getUB(), ub2 = re2.getUB(), ub = null;
		Relation lbrel = compare(lb1, rd1, lb2, rd2);
		Relation ubrel = compare(ub1, rd1, ub2, rd2);

		// Compare lower bounds and take MAX
		if ( lbrel.isGE() )
			lb = lb1;

		else if ( lbrel.isLT() )
			lb = lb2;

		else if ( range_accuracy < 1 )
			lb = lb1;

		else if ( range_accuracy < 2 )
		{
			// Check if re1 < re2.
			Relation rel = compare(ub1, rd1, lb2, rd2);

			// Return an invalid range if intersection is empty
			if ( rel.isLT() )
				return new RangeExpression(
					new IntegerLiteral(1), new IntegerLiteral(-1));

			else if ( rel.isEQ() || lb2 instanceof IntegerLiteral )
				lb = lb2;

			else
				lb = lb1;
		}

		else
			lb = simplify(new MinMaxExpression(false, lb1, lb2));

		// Compare upper bounds and take MIN
		if ( ubrel.isLE() )
			ub = ub1;

		else if ( ubrel.isGT() )
			ub = ub2;

		else if ( range_accuracy < 1 )
			ub = ub1;

		else if ( range_accuracy < 2 )
		{
			// Check if re1 < re2.
			Relation rel = compare(ub1, rd1, lb2, rd2);

			if ( rel.isLT() )
				return new RangeExpression(
					new IntegerLiteral(1), new IntegerLiteral(-1));

			else if ( ub2 instanceof IntegerLiteral )
				ub = ub2;

			else
				ub = ub1;
		}

		else
			ub = simplify(new MinMaxExpression(true, ub1, ub2));

		// Detect MAX(a,b):MIN(a,b); just return the first expression.
		if ( lb instanceof MinMaxExpression && ub instanceof MinMaxExpression &&
			!((MinMaxExpression)lb).isMin() && ((MinMaxExpression)ub).isMin() &&
			compareChildren(lb, ub) == 0 )
			return re1.getLB();

		if ( lb.compareTo(ub) == 0 )
			return lb;
		else
			return new RangeExpression(lb, ub);
	}


	/**
	 * Computes the union of the two expressions with the given range domains.
	 * @param e1			first expression
	 * @param rd1			first range domain
	 * @param e2			second expression
	 * @param rd2			second range domain
	 * @return				union of the two expressions
	 */
	public static Expression unionRanges
	(Expression e1, RangeDomain rd1, Expression e2, RangeDomain rd2)
	{
		// [lb:ub] = [a:b] U [c:d],    lb = a        if a<=c
		//                                = c        if a>c
		//                                = -INF     otherwise (range_accuracy=0)
		//                                = heuristics         (range_accuracy=1)
		//                                = min(a,c)           (range_accuracy=2) 
		//                             ub = b        if b>=d
		//                                = d        if b<d
		//                                = +INF     otherwise (range_accuracy=0)
		//                                = heuristics         (range_accuracy=1)
		//                                = max(b,d)           (range_accuracy=2)
		//
		// Check if either e1/e2 is omega range, empty range, or e1==e2.
		if ( isOmega(e1) || isOmega(e2) )
			return null;
		else if ( isEmpty(e1) )
			return (Expression)e2.clone();
		else if ( isEmpty(e2) )
			return (Expression)e1.clone();
		else if ( e1.compareTo(e2) == 0 )
			return (Expression)e1.clone();

		// Converts e1 & e2 to range expressions. re1 and re2 contain cloned copy
		// of e1 and e2 in their lb and ub.
		RangeExpression re1 = toRange(e1), re2 = toRange(e2);
		Expression lb1 = re1.getLB(), lb2 = re2.getLB(), lb = null;
		Expression ub1 = re1.getUB(), ub2 = re2.getUB(), ub = null;

		Relation lbrel = compare(lb1, rd1, lb2, rd2);
		Relation ubrel = compare(ub1, rd1, ub2, rd2);

		// Compare lower bounds and take MIN
		if ( lbrel.isLE() )
			lb = lb1;

		else if ( lbrel.isGT() )
			lb = lb2;

		else if ( range_accuracy < 1 )
			lb = new InfExpression(-1);

		// Sign comparison
		else if ( range_accuracy < 2 )
		{
			Relation rel = compare(ub1, rd1, lb2, rd2);
			if ( rel.isLE() )
			{
				rel = compare(lb1, rd1, ub2, rd2);
				if ( rel.isGT() )
					lb = new InfExpression(-1);
				else
					lb = lb1;
			}
			else
			{
				Expression zero = new IntegerLiteral(0);
				Relation sign1 = compare(lb1, rd1, zero, rd2);
				Relation sign2 = compare(lb2, rd2, zero, rd1);
				if ( sign1.isGE() && sign2.isGE() )
					lb = zero;
				else
					lb = new InfExpression(-1);
			}
		}

		else
			lb = simplify(new MinMaxExpression(true, lb1, lb2));

		// Compare upper bounds and take MAX
		if ( ubrel.isGE() )
			ub = ub1;

		else if ( ubrel.isLT() )
			ub = ub2;

		else if ( range_accuracy < 1 )
			ub = new InfExpression(1);

		else if ( range_accuracy < 2 )
		{
			Relation rel = compare(ub1, rd1, lb2, rd2);
			if ( rel.isLE() )
			{
				rel = compare(lb1, rd1, ub2, rd2);
				if ( rel.isGT() )
					ub = new InfExpression(1);
				else
					ub = ub2;
			}
			else
			{
				Expression zero = new IntegerLiteral(0);
				Relation sign1 = compare(ub1, rd1, zero, rd2);
				Relation sign2 = compare(ub2, rd2, zero, rd1);
				if ( sign1.isLE() && sign2.isLE() )
					ub = zero;
				else
					ub = new InfExpression(1);
			}
		}

		else
			ub = simplify(new MinMaxExpression(false, ub1, ub2));

		if ( lb.compareTo(ub) == 0 )
			return lb;
		else
			return new RangeExpression(lb,ub);
	}

	// Compute widening operation
	private static Expression widenRange
	(Expression e, Expression widen, RangeDomain rd)
	{
		// [lb:ub] = [a:b] W [c:d],    lb = a        if a=c
		//                                = -INF     otherwise
		//                             ub = b        if b=d
		//                                = +INF     otherwise
		//
		// Check if e1/e2 is empty or omega.
		if ( isOmega(e) || isOmega(widen) )
			return null;
		else if ( isEmpty(e) )
			return (Expression)widen.clone();
		else if ( isEmpty(widen) )
			return (Expression)e.clone();

		// Convert the two expressions to range expressions.
		RangeExpression re = toRange(e), rwiden = toRange(widen);

		// Compare lower bounds
		Relation rel = rd.compare(re.getLB(), rwiden.getLB());
		if ( !rel.isEQ() )
			re.setLB(new InfExpression(-1));

		// Compare upper bounds
		rel = rd.compare(re.getUB(), rwiden.getUB());
		if ( !rel.isEQ() )
			re.setUB(new InfExpression(1));

		if ( re.getLB().compareTo(re.getUB()) == 0 )
			return re.getLB();
		else
			return re;
	}

	// Compute narrowing operation
	private static Expression narrowRange
	(Expression e, Expression narrow, RangeDomain rd)
	{
		// [lb:ub] = [a:b] N [c:d],    lb = a        if a != -INF
		//                                = c        otherwise
		//                             ub = b        if b != +INF
		//                                = d        otherwise
		//
		// Check if operation is singular
		if ( isOmega(narrow) )
			return ( e==null )? null: (Expression)e.clone();
		else if ( isOmega(e) || !(narrow instanceof RangeExpression) )
			return (Expression)narrow.clone();

		// Convert the two expressions to range expressions.
		RangeExpression re = toRange(e), rnarrow = toRange(narrow);
		Expression lb = null, ub = null;

		if ( re.getLB() instanceof InfExpression )
			re.setLB(rnarrow.getLB());

		if ( re.getUB() instanceof InfExpression )
			re.setUB(rnarrow.getUB());

		if ( re.getLB().compareTo(re.getUB()) == 0 )
			return re.getLB();
		else
			return re;
	}


	/*==================================================================
	  Interfaces for tailored simplifier in NormalExpression
	  ==================================================================*/

	/**
   * Interfaces to the NormalExpression simplifier tailored for range analysis.
   * @param e			expression being simplified
   * @return			simplified expression
   */
	private static Expression simplify(Expression e)
	{
		Expression ret = NormalExpression.simplify(e, "RANGE");
		return NormalExpression.simplify(e, "RANGE");
	}

  /**
   * Adds two expressions and simplifies the result.
   * @param e1    first operand
   * @param e2    second operand
   * @return      result
   */
	private static Expression add(Expression e1, Expression e2)
	{
		return simplify(new BinaryExpression(e1, BinaryOperator.ADD, e2));
	}

  /**
   * Subtracts an expression from the other and simplifies the result.
   * @param e1    first operand
   * @param e2    second operand
   * @return      result
   */
	private static Expression subtract(Expression e1, Expression e2)
	{
		return simplify(new BinaryExpression(e1, BinaryOperator.SUBTRACT, e2));
	}

  /**
   * Multiplies an expression by the other and simplifies the result.
   * @param e1    first operand
   * @param e2    second operand
   * @return      result
   */
	private static Expression multiply(Expression e1, Expression e2)
	{
		return simplify(new BinaryExpression(e1, BinaryOperator.MULTIPLY, e2));
	}

  /**
   * Divides an expression by the other and simplifies the result.
   * @param e1    first operand
   * @param e2    second operand
   * @return      result
   */
	private static Expression divide(Expression e1, Expression e2)
	{
		return simplify(new BinaryExpression(e1, BinaryOperator.DIVIDE, e2));
	}


	/*====================================================================
    Miscellaneous helper methods
    ====================================================================*/

	// Return a new omega expression.
	private static Expression omega()
	{
		return
			new RangeExpression(new InfExpression(-1), new InfExpression(1));
	}


	private static String SymbolstoString(Collection c)
	{
		String ret = "{";

		int i = 0;
		for ( Object o : c )
		{
			if ( i++ > 0 )
				ret += ", ";
			if ( o instanceof Symbol )
				ret += ((Symbol)o).getSymbolName();
			else
				ret += o;
		}
		return ret+"}";
	}


	// Converts min/max expression to conditional expression.
	private static void removeMinMax(Expression e)
	{
		if ( e == null )
			return;

		FlatIterator iter = new FlatIterator(e);

		while ( iter.hasNext() )
			removeMinMax((Expression)iter.next());

		if ( e instanceof MinMaxExpression )
			e.swapWith(((MinMaxExpression)e).toConditionalExpression());
	}


	// Convert an expression to a range expression
	//private static RangeExpression toRange(Expression e)
	public static RangeExpression toRange(Expression e)
	{
		if ( e instanceof RangeExpression )
			return (RangeExpression)e.clone();
		return new RangeExpression((Expression)e.clone(), (Expression)e.clone());
	}


	// Test if an expression is omega
	private static boolean isOmega(Expression e)
	{
		if ( e == null || e instanceof InfExpression )
			return true;
		else if ( e instanceof RangeExpression )
			return ((RangeExpression)e).isOmega();
		else
			return false;
	}


	// Test if an expression is too complex to be in the range
	private static boolean isDiscarded(Expression e)
	{
		// Test 1: Class-based filtering
		DepthFirstIterator iter = new DepthFirstIterator(e);

		Set discard_set = new HashSet(Arrays.asList(
			CharLiteral.class,
			FloatLiteral.class,
			StringLiteral.class,
			AccessExpression.class,
			CommaExpression.class,
			ConditionalExpression.class,
			FunctionCall.class,
			VaArgExpression.class,
			SizeofExpression.class,
			Typecast.class
		));
			
		while ( iter.hasNext() )
			if ( discard_set.contains(iter.next().getClass()) )
				return true;

		// Test 2: Pattern-based filtering
		iter.reset();
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof MinMaxExpression )
			{
				MinMaxExpression me = (MinMaxExpression)o;
				for ( Object child : me.getChildren() )
					if ( child instanceof MinMaxExpression &&
						me.isMin() != ((MinMaxExpression)child).isMin() )
						return true;
			}
		}

		return false;
	}


	// Test if an expression is numerically empty
	private static boolean isEmpty(Expression e)
	{
		return ( e instanceof RangeExpression && ((RangeExpression)e).isEmpty() );
	}


	// Test if an expression is symbolically empty
	private static boolean isEmpty(Expression e, RangeDomain rd)
	{
		if ( isEmpty(e) )
			return true;
		if ( !(e instanceof RangeExpression) )
			return false;
		RangeExpression re = (RangeExpression)e;
		Relation rel = rd.compare(re.getLB(), re.getUB());
		return ( rel.isGT() );
	}


	public boolean isEmptyRange(Expression e)
	{
		if ( isEmpty(e) )
			return true;
		if ( !(e instanceof RangeExpression) )
			return false;
		RangeExpression re = (RangeExpression)e;
		Relation rel = compare(re.getLB(), re.getUB());
		return ( rel.isGT() );
	}


	public boolean encloses(Expression e1, Expression e2)
	{
		RangeExpression re1 = toRange(e1);
		RangeExpression re2 = toRange(e2);

		return (
			compare(re1.getLB(), re2.getLB()).isLE() &&
			compare(re1.getUB(), re2.getUB()).isGE()
		);
	}


	// Compare equality of the children of the two expressions.
	private static int compareChildren(Expression e1, Expression e2)
	{
		List child1 = e1.getChildren(), child2 = e2.getChildren();

		if ( child1.size() != child2.size() )
			return (child1.size()>child2.size())? 1: -1;

		for ( int i=0,ret=0; i<child1.size(); ++i )
		{
			ret = ((Expression)child1.get(i)).compareTo((Expression)child2.get(i));
			if ( ret != 0 )
				return ret;
		}
		return 0;
	}


	// Global comparison cache for wide-window history.
	private static class ComparisonCache extends LinkedHashMap
	{
		private int MAX_ENTRIES;

		public ComparisonCache()
		{
			super();
		}

		public ComparisonCache(int max_entries)
		{
			super();
			MAX_ENTRIES = max_entries;
		}

		protected boolean removeEldestEntry(Map.Entry eldest)
		{
			return size() > MAX_ENTRIES;
		}

	}

}
