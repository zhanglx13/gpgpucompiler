package cetus.analysis;

import java.util.*;
import cetus.hir.*;
import cetus.exec.*;

/**
 * RangeAnalysis performs symbolic range propagation for the programs by
 * symbolically executing the program and abstracting the values of integer
 * variables with a symbolic bounds. The implementation is based on Symbolic
 * Range Propagation by Blume and Eigenmann, which was implemented for the
 * Fortran77 language.
 * <p>
 * Range analysis returns a map from each statement to its corresponding
 * {@link RangeDomain} that contains the set of valid value ranges before
 * executing the statement. The result of this analysis was verified with the
 * C benchmarks in the SPEC CPU2006 suite. Range analysis does not handle
 * integer overflow as it does not model the overflow but such cases are rare
 * in normal correct programs. Following example shows how to invoke a range
 * analyzer in a compiler pass:
 * <p>
 * <code>
 * RangeAnalysis analyzer = new RangeAnalysis(program); <br>
 * Map range_map = analyzer.getRangeMap(procedure); <br>
 * ... <br>
 * RangeDomain range_domain = range_map.get(statement); <br>
 * // range_domain now contains the set of value ranges for the statement. <br>
 * </code>
 */
public class RangeAnalysis extends AnalysisPass
{
	// Pass tag
	private static final String tag = "[RangeAnalysis]";

	// Tractable classes.
	private static final Set tractable_class =
	new HashSet(Arrays.asList(
		BinaryExpression.class,
		Identifier.class,
		InfExpression.class,
		IntegerLiteral.class,
		MinMaxExpression.class,
		RangeExpression.class,
		UnaryExpression.class
	));

	// Tractable operators.
	private static final Set tractable_op =
	new HashSet(Arrays.asList(
		BinaryOperator.ADD,
		BinaryOperator.DIVIDE,
		BinaryOperator.MODULUS,
		BinaryOperator.MULTIPLY,
		UnaryOperator.MINUS,
		UnaryOperator.PLUS
	));

	// Analyzed range map.
	//private Map range_map;

	// Debug level
	private static int debug;

	// Validation flag
	private int validation;

	/**
	 * Constructs a range analyzer for the program.
	 *
	 * @param program	the input program.
	 */
	public RangeAnalysis(Program program)
	{
		super(program);

		// Debug information
		debug = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
		// Range Validation
		try {
			validation = new Integer(System.getenv("CETUS_RANGE_VALIDATION"));
		} catch ( Exception ex ) {
			validation = 0;
		}
	}

	/**
	 * Returns the pass name.
	 *
	 * @return the pass name in string.
	 */
	public String getPassName()
	{
		return tag;
	}

	/**
	 * Starts range analysis for the whole program. this is useful only for
	 * testing the range analysis.
	 */
	public void start()
	{
		double timer = Tools.getTime();

		DepthFirstIterator iter = new DepthFirstIterator(program);
		iter.pruneOn(Procedure.class);
		iter.pruneOn(Statement.class);
		iter.pruneOn(Declaration.class);

		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Procedure )
				getRangeMap((Procedure)o);
		}
		
		timer = Tools.getTime(timer);

		Tools.printStatus("[RangeAnalysis Time] "+String.format("%.2f\n",timer),1);

		// Add #include <assert.h>
		if ( validation > 0 )
			addAssertHeader();
	}

	/**
	 * Starts range analysis for the procedure.
	 *
	 * @param proc		the input procedure.
	 * @return the range map for the procedure.
	 */
	public Map getRangeMap(Procedure proc)
	{
		double timer = Tools.getTime();
		Tools.printStatus("Range analysis of " + proc.getName() + " ... ", 1);

		// Builds a directed graph for the procedure
		CFGraph g = new CFGraph(proc);

		// Expands the graph whose node has at most one assignment
		g.normalize();

		if ( debug >= 5 )
		{
			Tools.printlnStatus("CFG of "+proc.getName(), 5);
			Tools.printlnStatus(g.toDot(), 5);
		}

		// Computes the topological order of the node disregarding back edges
		g.topologicalSort(g.getNodeWith("stmt","ENTRY"));

		// Widening phase of fixpoint iteration
		iterateToFixpoint(g, true);

		if ( debug >= 5 )
		{
			Tools.printlnStatus("CFG after widening", 5);
			Tools.printlnStatus(g.toDot("top-order,ranges,ir,tag", 3), 5);
			Tools.printlnStatus("Range Map after widening", 5);
			Tools.printlnStatus(
				toPrettyRanges(proc, getRangeMap(g), new Integer(0)), 5);
		}

		// Narrowing phase of fixpoint iteration
		iterateToFixpoint(g, false);

		Map range_map = getRangeMap(g);

		if ( debug >= 5 )
		{
			Tools.printlnStatus("CFG after narrowing", 5);
			Tools.printlnStatus(g.toDot("top-order,ranges,ir,tag", 3), 5);
			Tools.printlnStatus("Range Map after narrowing", 5);
			Tools.printlnStatus(
				toPrettyRanges(proc, range_map, new Integer(0)), 5);
		}

		// Add assert() call for runtime verification
		if ( validation > 0 )
			addAssertCall(proc, range_map);

		Tools.printlnStatus(String.format("%.2f", Tools.getTime(timer)) +
			" seconds.", 1);

		return range_map;
	}

	/**
   * Generates a range map from the work space graph.
   *
   * @param g     the control flow graph used in the analysis.
   * @return      the map from statements to sets of value ranges.
   */
	private static Map getRangeMap(CFGraph g)
	{
		Map ret = new HashMap();
		for ( int i=0; i < g.size(); ++i )
		{
			DFANode node = g.getNode(i);
			Object stmt = node.getData("stmt");
			RangeDomain rd = (RangeDomain)node.getData("ranges");
			if ( stmt == null || !(stmt instanceof Traversable) || rd == null )
				continue;

			// Does not include out-of-scope ranges with same name.
			Set<Symbol> vars = new HashSet(rd.getSymbols());
			for ( Symbol var : vars )
			{
				// Test with key
				IDExpression id = new Identifier(var.getSymbolName());

				if ( !isValidScope((Traversable)stmt, id, var) )
				{
					rd.removeRange(var);
					continue;
				}

				// Test with data
				DepthFirstIterator iter = new DepthFirstIterator(rd.getRange(var));

				while ( iter.hasNext() )
				{
					Object o = iter.next();
					if ( o instanceof Identifier )
					{
						Identifier ident = (Identifier)o;
						if ( !isValidScope((Traversable)stmt, ident, ident.getSymbol()) )
						{
							rd.removeRange(var);
							break;
						}
					}
				}
			}

			// Eliminate variables with pointer type.
			vars = new HashSet(rd.getSymbols());
			for ( Symbol var : vars )
				if ( Tools.isPointer(var) || containsPointerType(rd.getRange(var)) )
					rd.removeRange(var);
				
			ret.put(stmt, rd);
		}

		return ret;
	}

	/**
   * Returns the range map in a pretty format.
   *
   * @param t        the traversable object.
   * @param ranges   the range map.
   * @param indent   the indent for pretty printing.
   * @return         the string in a pretty format.
   */
	public static String toPrettyRanges
	(Traversable t, Map ranges, Integer indent)
	{
		String ret = "", tab = "";

		for ( int i=0; i<indent; ++i )
			tab += "  ";

		if ( t instanceof Procedure )
		{
			Procedure p = (Procedure)t;
			ret += "Range Domain for Procedure "+p.getName()+"\n";
			ret += toPrettyRanges(p.getBody(), ranges, indent);
		}

		else if ( t instanceof CompoundStatement )
		{
			ret += tab+"{\n";
			indent++;

			for ( Object o: t.getChildren() )
				ret += toPrettyRanges((Traversable)o, ranges, indent);

			indent--;
			ret += tab+"}\n";
		}

		else if ( t instanceof DoLoop )
		{
			DoLoop d = (DoLoop)t;
			ret += tab+"do\n";
			ret += toPrettyRanges(d.getBody(), ranges, indent);
			ret += tab+"while ( "+d.getCondition()+" );\n";
		}

		else if ( t instanceof ForLoop )
		{
			ForLoop f = (ForLoop)t;
			ret += tab+"for ( ";
			Statement init = f.getInitialStatement();
			ret += ((init==null)? ";": init)+" ";
			Expression condition = f.getCondition();
			ret += ((condition==null)? " ": condition)+"; ";
			Expression step = f.getStep();
			ret += ((step==null)? "": step)+" )\n";
			ret += toPrettyRanges(f.getBody(), ranges, indent);
		}

		else if ( t instanceof IfStatement )
		{
			IfStatement i = (IfStatement)t;
			ret += tab+"if ( "+i.getControlExpression()+" )\n";
			ret += toPrettyRanges(i.getThenStatement(), ranges, indent);
			Statement els = i.getElseStatement();
			if ( els != null )
				ret += tab+"else\n"+toPrettyRanges(els, ranges, indent);
		}

		else if ( t instanceof SwitchStatement )
		{
			SwitchStatement s = (SwitchStatement)t;
			ret += tab+"switch ( "+s.getExpression()+" )\n";
			ret += toPrettyRanges(s.getBody(), ranges, indent);
		}

		else if ( t instanceof WhileLoop )
		{
			WhileLoop w = (WhileLoop)t;
			ret += tab+"while ( "+w.getCondition()+" )\n";
			ret += toPrettyRanges(w.getBody(), ranges, indent);
		}

		else if ( t instanceof Statement )
			ret += tab+t+"\n";


		if ( t instanceof Statement )
		{
			RangeDomain rd = (RangeDomain)ranges.get(t);
			ret = "        "+((rd==null)? "[]": rd)+"\n"+ret;
		}
		return ret;
	}

	/**
	 * Returns a range domain for the given statement.
	 *
	 * @param stmt the input statement.
	 * @return the out range domain for the statement.
	 */
	public static RangeDomain getRangeDomain(Statement stmt)
	{
		if ( stmt == null || !(stmt instanceof ExpressionStatement) )
			return new RangeDomain();

		CFGraph cfg = new CFGraph(stmt);
		cfg.normalize();
		cfg.topologicalSort(cfg.getNodeWith("stmt","ENTRY"));
		DFANode last = cfg.getNodeWith("top-order", new Integer(cfg.size()-1));
		DFANode exit = new DFANode("stmt","EXIT");
		exit.putData("top-order", new Integer(cfg.size()));
		cfg.addEdge(last, exit);
		// Exit node is necessary because Range Analysis computes only in-range.
		// Maybe CFGraph needs flow exit always but it breaks the range analysis,
		// so it needs to be revisited later.

		iterateToFixpoint(cfg, true);

		RangeDomain ret = (RangeDomain)exit.getData("ranges");
		if ( ret == null )
			ret = new RangeDomain();

		return ret;
	}

	// Add #include <assert.h>
	private void addAssertHeader() throws InternalError
	{
		DepthFirstIterator iter = new DepthFirstIterator(program);
		iter.pruneOn(TranslationUnit.class);
		
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof TranslationUnit )
				((TranslationUnit)o).setHeader("#include <assert.h>\n\n");
		}
	}


	// Check if the identifier has a valid scope.
	private static boolean isValidScope
	(Traversable from, IDExpression id, Symbol symbol)
	{
		SymbolTable symtab = null, proc = null;
		Traversable t = from.getParent();
		while ( t != null )
		{
			if ( t instanceof SymbolTable )
			{
				symtab = (SymbolTable)t;
				proc = null;
				if ( symtab.getTable().containsKey(id) )
					break;
				// Formal parameters are orphans.
				if ( t.getParent() instanceof Procedure )
				{
					proc = (SymbolTable)t.getParent();
					if ( proc.getTable().containsKey(id) )
						break;
				}
			}
			t = t.getParent();
		}
		
		return (
			Tools.getSymbols(symtab).contains(symbol) ||
			Tools.getSymbols(proc).contains(symbol)
		);
	}


	// Eliminates variables with pointer type.
	private static boolean containsPointerType(Traversable t)
	{
		if ( t == null )
			return false;

		DepthFirstIterator iter = new DepthFirstIterator(t);
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Identifier &&
			Tools.isPointer(((Identifier)o).getSymbol()) )
				return true;
		}
		return false;
	}


	// Convert a range domain to an assert function call.
	private static Statement toAssertCall(RangeDomain rd)
	{
		Expression arg = rd.toExpression();

		if ( arg == null )
			return null;

		FunctionCall fcall = new FunctionCall(new Identifier("assert"));
		fcall.addArgument(arg);

		return new ExpressionStatement(fcall);
	}

	// Add assertion before statements.
	private static void addAssertCall(Procedure proc, Map ranges)
	{
		DepthFirstIterator iter = new DepthFirstIterator(proc);

		while ( true )
		{
			Statement stmt = null;

			try {
				stmt = (Statement)iter.next(Statement.class);
			} catch ( NoSuchElementException ex ) {
				break;
			}

			Traversable parent = stmt;
			while ( !(parent instanceof CompoundStatement) )
				parent = parent.getParent();

			RangeDomain rd = (RangeDomain)ranges.get(stmt);
			if ( rd == null )
				continue;

			Statement assert_call = toAssertCall((RangeDomain)ranges.get(stmt));
			if ( assert_call == null )
				continue;

			// While loop may have a non-compound child, IllegalArgumentException
			// being thrown.
			try {
				((CompoundStatement)parent).addStatementBefore(stmt, assert_call);
			} catch ( IllegalArgumentException ex ) {
				;
			}

		}
	}


	// Marks the node having a back edge by comparing topological order
	private static void setBackedge(DFANode node)
	{
		int my_order = (Integer)node.getData("top-order");
		for ( DFANode pred : node.getPreds() )
			if ( my_order < (Integer)pred.getData("top-order") )
			{
				node.putData("has-backedge", new Boolean(true));
				break;
			}
	}


	// Fixpoint iteration
	private static void iterateToFixpoint(CFGraph g, boolean widen)
	{
		TreeMap work_list = new TreeMap();

		// Add the entry node to the work list for widening phase.
		if ( widen )
		{
			DFANode entry = g.getNodeWith("stmt", "ENTRY");
			entry.putData("ranges", new RangeDomain());
			work_list.put(entry.getData("top-order"), entry);
		}

		// Add the widened nodes to the work list for narrowing phase.
		else
		{
			for ( int i=0; i<g.size(); ++i )
			{
				DFANode widened = g.getNode(i);
				if ( widened.getData("has-backedge") != null )
					work_list.put(widened.getData("top-order"), widened);
			}
		}

		while ( work_list.size() > 0 )
		{
			// Get the first node in topological order from the work list.
			DFANode node = (DFANode)work_list.remove(work_list.firstKey());

			// Record number of iterations for each node.
			Integer visits = (Integer)node.getData("num-visits");
			if ( visits == null )
			{
				node.putData("num-visits", new Integer(1));
				setBackedge(node);
			}
			else
				node.putData("num-visits", new Integer(visits+1));

			// Merge incoming states from predecessors.
			RangeDomain curr_ranges = null;
			for ( DFANode pred : node.getPreds() )
			{
				RangeDomain pred_range_out = (RangeDomain)node.getPredData(pred);

				// Skip BOT-state predecessors that has not been visited.
				if ( pred_range_out == null )
					continue;

				if ( curr_ranges == null )
					curr_ranges = new RangeDomain(pred_range_out);
				else
					curr_ranges.unionRanges(pred_range_out);

			}

			// Add initial values from declarations
			enterScope(node, curr_ranges);

			// Widening/Narrowing operations.
			RangeDomain prev_ranges = (RangeDomain)node.getData("ranges");
			if ( prev_ranges != null && node.getData("has-backedge") != null )
			{
				if ( widen )
				{
					Set widener = (Set)node.getData("loop-variants");

					// Selective widening only with loop-variant symbols.
					if ( widener != null && widener.size() > 0 )
						curr_ranges.widenAffectedRanges(prev_ranges, (Set<Symbol>)widener);

					else
						curr_ranges.widenRanges(prev_ranges);
				}
				else
					curr_ranges.narrowRanges(prev_ranges);
			}

			if ( prev_ranges == null || !prev_ranges.equals(curr_ranges) )
			{
				node.putData("ranges", curr_ranges);

				// Apply state changes due to the execution of the node.
				updateRanges(node);

				// Clean up after exiting a scope.
				exitScope(node);

				for ( DFANode succ : node.getSuccs() )
					// Do not add successors for infeasible paths
					if ( succ.getPredData(node) != null )
						work_list.put(succ.getData("top-order"), succ);
			}
		}
	}


	// Add intialized values from the declarations.
	private static void enterScope(DFANode node, RangeDomain ranges)
	{
		SymbolTable st = (SymbolTable)node.getData("symbol-entry");
		if ( st == null )
			return;

		for ( Symbol var : Tools.getSymbols(st) )
		{
			// Not interested in other declarators.
			if ( !(var instanceof VariableDeclarator) )
				continue;

			Initializer init = ((VariableDeclarator)var).getInitializer();

			// Flow of initializer is not guaranteed.
			if ( Tools.containsSideEffect(init) )
			{
				ranges.clear();
				return;
			}

			// Extract only scalar integers' initial values.
			List specs = var.getTypeSpecifiers();
			if (
				init == null ||
				init.getChildren().size() != 1 || // #initializer must be one.
				!Tools.isScalar(var) ||
				!isTractableType(var) ||
				specs.contains(Specifier.VOLATILE) ||
				specs.contains(Specifier.STATIC)
			) continue;

			Expression new_range = 
				NormalExpression.simplify((Expression)init.getChildren().get(0));

			if ( isTractableRange(new_range) )
				ranges.setRange(var, new_range);
		}
	}


	// Clean up variables that are not present in the scope.
	private static void exitScope(DFANode node)
	{
		List symbol_exits = (List)node.getData("symbol-exit");

		if ( symbol_exits != null )
		{
			Set symbols = new HashSet();

			// symbol-exit assumed to be list of symbol tables
			for ( Object o : symbol_exits )
				symbols.addAll(Tools.getSymbols((SymbolTable)o));

			for ( DFANode succ : node.getSuccs() )
			{
				RangeDomain ranges_out = (RangeDomain)succ.getPredData(node);

				if ( ranges_out != null )
					ranges_out.removeSymbols(symbols);
			}
		}
	}


	// Check if the expression is tractable by range analysis.
	private static boolean isTractableRange(Expression e)
	{
		DepthFirstIterator iter = new DepthFirstIterator(e);
		while ( iter.hasNext() )
		{
			Object o = iter.next();

			if ( !tractable_class.contains(o.getClass()) )
				return false;
			if ( o instanceof Identifier )
			{
				Identifier id = (Identifier)o;
				if ( !isTractableType(id.getSymbol()) )
					return false;
			}
			else if ( o instanceof BinaryExpression &&
				!tractable_op.contains(((BinaryExpression)o).getOperator()) )
				return false;
			else if ( o instanceof UnaryExpression &&
				!tractable_op.contains(((UnaryExpression)o).getOperator()) )
				return false;
			// Other cases should be all tractable.
		}
		return true;
	}


	// Check if the symbol is tractable by range analysis.
	private static boolean isTractableType(Symbol symbol)
	{
		return ( Tools.isInteger(symbol) &&
			!symbol.getTypeSpecifiers().contains(Specifier.UNSIGNED) );
	}


	// Detect invertible assignment and return the inverted expression.
	private static Expression invertExpression(Expression to, Expression from)
	{
		Symbol var = Tools.getSymbolOf(to);

		Expression diff = NormalExpression.subtract(to, from);

		if ( Tools.containsSymbol(diff, var) )
			return null;

		return NormalExpression.add(to, diff);
	}


	// Methods for updating edges to successors.
	private static void updateRanges(DFANode node)
	{
		//Object o = node.getData(data_keys);
		Object o = CFGraph.getIR(node);

		if ( o instanceof ExpressionStatement )
			o = ((ExpressionStatement)o).getExpression();

		// Side-effect node
		if ( o instanceof Traversable &&
			(Tools.containsClass((Traversable)o, FunctionCall.class) ||
			Tools.containsClass((Traversable)o, VaArgExpression.class)) )
		{
			updateUnsafeNode(node);
			return;
		}

		// Assignments
		if ( o instanceof AssignmentExpression )
			updateAssignment(node, (AssignmentExpression)o);

		// Binary expressions (logical expression)
		else if ( o instanceof BinaryExpression )
			updateBinary(node, (BinaryExpression)o);

		// Switch statements
		else if ( o instanceof SwitchStatement )
			updateSwitch(node, (SwitchStatement)o);

		// Side-effect-free node
		else
			updateSafeNode(node);
	}


	// Side-effect-present node.
	private static void updateUnsafeNode(DFANode node)
	{
		for ( DFANode succ : node.getSuccs() )
			succ.putPredData(node, new RangeDomain());
	}


	// Side-effect-free node.
	private static void updateSafeNode(DFANode node)
	{
		RangeDomain ranges_in = (RangeDomain)node.getData("ranges");

		for ( DFANode succ : node.getSuccs() )
			succ.putPredData(node, new RangeDomain(ranges_in));
	}


	// Update assignments.
	private static void updateAssignment(DFANode node, AssignmentExpression e)
	{
		// Compute invariant information and cache it.
		if ( (Integer)node.getData("num-visits") < 2 )
		{
			Expression to = e.getLHS();
			Expression from = NormalExpression.simplify(e.getRHS());
			String direction = "nochange";

			// Case 1. Dereference is treated conservatively.
			if ( Tools.containsUnary(to, UnaryOperator.DEREFERENCE) )
				direction = "discard";

			else if ( to instanceof Identifier )
			{
				Symbol var = ((Identifier)to).getSymbol();

				// Case 1. Unknown types are treated conservatively.
				if ( var == null )
					direction = "discard";

				// Case 2. Lvalue is not tractable by range analysis.
				else if ( !isTractableType(var) )
					direction = "nochange";

				// Case 3. Lvalue is tractable but range is not.
				else if ( !isTractableRange(from) )
					direction = "kill";

				// Case 4. Typical assignment with no self dependence.
				else if ( !Tools.containsSymbol(from, var) )
					direction = "normal";

				else 
				{
					Expression inverted = invertExpression(to, from);

					// Case 5. There is a self dependence, and it is not invertible.
					if ( inverted == null )
						direction = "recurrence";

					// Case 6. There is a self dependence, and it is invertible.
					else
					{
						direction = "invertible";
						node.putData(direction, inverted);
					}
				}
				node.putData("assign-to", to);
				node.putData("assign-from", from);
			}

			// Case 7. Lvalue is not simple; no source of information.
			node.putData("assign-direction", direction);
		}

		RangeDomain ranges_in = (RangeDomain)node.getData("ranges");

		RangeDomain ranges_out = new RangeDomain(ranges_in);

		String direction = (String)node.getData("assign-direction");

		if ( direction.equals("discard") )
			ranges_out = new RangeDomain();

		else if ( !direction.equals("nochange") )
		{
			Symbol var = ((Identifier)node.getData("assign-to")).getSymbol();

			Expression replace_with = ( direction.equals("invertible") )?
				(Expression)node.getData("invertible"): ranges_out.getRange(var);

			Expression from = (Expression)node.getData("assign-from");

			// Preprocess the range to minimize the chance of being cyclic.
			if ( direction.equals("invertible") || direction.equals("recurrence") )
			{
				ranges_out.forwardSubstitute();
				from = ranges_out.replaceSymbol(
					(Expression)from.clone(), var, ranges_out.getRange(var));
			}

			// Eliminate the assigned symbol in the range.
			ranges_out.replaceSymbol(var, replace_with);

			// Postprocess the range by discarding cyclic ranges.
			ranges_out.removeRecurrence();

			// Remove or keep the range for the assigned symbol.
			if ( direction.equals("kill") )
				ranges_out.removeRange(var);
			else
				ranges_out.setRange(var, from);
		}

		// Update successors.
		for ( DFANode succ : node.getSuccs() )
			succ.putPredData(node, ranges_out);
	}


	// Apply conditional expressions.
	private static void updateBinary(DFANode node, BinaryExpression e)
	{
		// Adjustment for unexpanded expression due to may-be evaluated expression.
		// Only safe evaluation is taken out of any expressions used in conditional
		// branches, and there should be a checking mechanism that detects unsafe
		// evaluations.
		if ( Tools.containsSideEffect(e) )
		{
			updateUnsafeNode(node);
			return;
		}
		// This is not a branch; residue of normalization.
		else if ( node.getSuccs().size() < 2 )
		{
			updateSafeNode(node);
			return;
		}

		// Compute or retrieve the ranges from the conditional expression.
		if ( (Integer)node.getData("num-visits") < 2 )
		{
			RangeDomain true_range = extractRanges(e);
			RangeDomain false_range = extractRanges(NormalExpression.negate(e));

			node.putSuccData((DFANode)node.getData("true"), true_range);
			node.putSuccData((DFANode)node.getData("false"), false_range);
		}

		RangeDomain ranges_in = (RangeDomain)node.getData("ranges");

		for ( DFANode succ : node.getSuccs() )
		{
			RangeDomain ranges_out = new RangeDomain(ranges_in);

			ranges_out.intersectRanges((RangeDomain)node.getSuccData(succ));

			// Infeasible path detected, so remove the data
			if ( ranges_in.size() > ranges_out.size() )
				ranges_out = null;

			succ.putPredData(node, ranges_out);
		}
	}


	// Apply conditional expressions from switch statement.
	private static void updateSwitch(DFANode node, SwitchStatement s)
	{
		Expression lhs = s.getExpression();
		if ( Tools.containsSideEffect(lhs) )
		{
			updateUnsafeNode(node);
			return;
		}

		// Compute and cache the extracted ranges. 
		if ( (Integer)node.getData("num-visits") < 2 )
		{
			for ( DFANode succ : node.getSuccs() )
			{
				//Object o = succ.getData(data_keys);
				Object o = CFGraph.getIR(node);

				if ( o instanceof Case )
					node.putSuccData(succ, 
						extractRanges(NormalExpression.eq(lhs,((Case)o).getExpression())));
			}
		}

		RangeDomain ranges_in = (RangeDomain)node.getData("ranges");

		for ( DFANode succ : node.getSuccs() )
		{
			RangeDomain ranges_out = new RangeDomain(ranges_in);

			RangeDomain edge_range = (RangeDomain)node.getSuccData(succ);

			if ( edge_range != null )
				ranges_out.intersectRanges(edge_range);

			// Infeasible path detected, so remove the data
			if ( ranges_in.size() > ranges_out.size() )
				ranges_out = null;

			succ.putPredData(node, ranges_out);
		}
	}


	// Extract a set of ranges from the preprocessed list
	private static RangeDomain extractRanges(List solved)
	{
		RangeDomain ret = new RangeDomain();
		for ( Object o : solved )
		{
			if ( !(o instanceof BinaryExpression) )
				continue;

			BinaryExpression be = (BinaryExpression)o;
			BinaryOperator op = be.getOperator();
			Expression lhs = be.getLHS();
			Expression rhs = NormalExpression.simplify(be.getRHS());

			Symbol var = ((Identifier)lhs).getSymbol();

			if ( !(lhs instanceof Identifier) ||
				!isTractableType(var) ||
				!isTractableRange(rhs) )
				continue;

			if ( op == BinaryOperator.COMPARE_EQ )
				ret.setRange(var, (Expression)rhs.clone());

			else if ( op == BinaryOperator.COMPARE_GE )
				ret.setRange(var, new RangeExpression(
					NormalExpression.simplify(rhs),
					new InfExpression(1)));

			else if ( op == BinaryOperator.COMPARE_GT )
				ret.setRange(var, new RangeExpression(
					NormalExpression.add(rhs, new IntegerLiteral(1)),
					new InfExpression(1)));

			else if ( op == BinaryOperator.COMPARE_LE )
				ret.setRange(var, new RangeExpression(
					new InfExpression(-1),
					NormalExpression.simplify(rhs)));

			else if ( op == BinaryOperator.COMPARE_LT )
				ret.setRange(var, new RangeExpression(
					new InfExpression(-1),
					NormalExpression.subtract(rhs, new IntegerLiteral(1))));

			else if ( op == BinaryOperator.COMPARE_NE )
				; // too wide
			else
				; // nothing to be done
		}
		return ret;
	}


	// Extract a set of ranges from the given binary expression
	private static RangeDomain extractRanges(Expression e)
	{
		RangeDomain ret = new RangeDomain();

		Expression simp = NormalExpression.simplify(e);

		if ( !(simp instanceof BinaryExpression) )
			return ret;

		BinaryExpression be = (BinaryExpression)simp;

		List solved = NormalExpression.solveForVariables(be);

		BinaryOperator op = be.getOperator();

		if ( op == BinaryOperator.LOGICAL_AND )
			for ( Object o : solved )
				ret.intersectRanges(extractRanges((List)o));

		else if ( op == BinaryOperator.LOGICAL_OR )
			for ( Object o : solved )
				ret.unionRanges(extractRanges((List)o));

		else
			ret = extractRanges(solved);

		return ret;
	}

}
