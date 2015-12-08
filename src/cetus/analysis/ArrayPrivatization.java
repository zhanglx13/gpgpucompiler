package cetus.analysis;

import java.util.*;

import cetus.hir.*;

import cetus.exec.*;

/*
 Performs array and scalar privatization analysis on all loops within a
 Program. Each instance of this analysis computes the must-defined set of
 variables and upward-exposed use set of variables for one iteration of each
 loop. These sets are aggregated when the instance is called from the outer
 loop analysis. The data contents for the array variables consist of mapping
 from variables to their corresponding subscript anges and the privatizable
 set is computed by set difference between Def and UEUse.
 
 The analysis is performed outwards starting from the inner-most loop
 structures and we use hierarchical control flow graph to ease this process.
 For each node in the control flow graph, the incoming Def is the solution
 of a forward data flow problem that finds the reaching definitions. We use
 the range analysis framework to enable subscript intersection operation at
 data flow join.

 The following procedure describes the high-level algorithm for array
 privatization.

 procedure Privatization(L)

   Input : Loop L
   Output: DEF[L], UEU[L], PRI[L]
   // DEF: Definded set
   // USE: Used set
   // UEU: Upward-exposed set
   // PRI: Private variables
   // (^): intersection, (v): union

   foreach direct inner loop L' in L
     (DEF[L'],USE[L']) = Privatization(L')

   G(N,E) = BuildCFG(L)
   // BuildCFG builds a CFG with the inner loops represented as super nodes

   Iteratively solve data flow equation DEF for node n in N
     DEFin[n] = (^) {DEFout[p], p in predecessors of n}
     DEFout[n] = (DEFin[n]-KILL[n]) (v) DEF[n]

   DEF[L] = {}
   UEU[L] = {}
   PRI[L] = CollectCandidates(L)
   // CollectCandidates collects arrays with loop-invariant subscripts

   foreach loop-exiting node e in N
     DEF[L] = DEF[L] (^) DEFout[e]

   foreach node n in N
     UEU[L] = UEU[L] (v) (USE[n]-DEFin[n])

   foreach variable v in UEU[L]
     PRI[L] = PRI[L]-{v}

   *Remove variables affected by function calls from the private list.
   *Remove user-typed variables from the private list.

   DEF[L] = AggregateDEF(DEF[L])
   UEU[L] = AggregateUSE(UEU[L])
   // AggregateDEF aggregates array sections in DEF (MUST set)
   // AggregateUSE aggregates array sections in USE (MAY set)

   return (DEF[L], UEU[L])

 end
 
 Other considerations.
 - Variable defined inside a loop is private to the loop
 - Aggregation process for non-canonical loops

 Required utilities
 - Operations for set of symbolic intervals
 - CFG with collapsed inner loops

 Current limitations
 1. All variables used in function calls and global variables are not listed
    in the private set if the analyzed loop contains any function calls.
 2. User-defined types are not listed in the private list.
 3. Incorporation of alias analysis is ongoing.
 4. There should be a way of detecting the inner loop's execution condition.
 */


/**
 * ArrayPrivatization performs array privatization analysis for each loop.
 * This analysis pass analyzes each loop in the program from inside to outside
 * while collecting array data flow information such as MUST-DEF and MAY-USE
 * and decides whether a certain variable is private based on the gathered
 * information. We use unit-strided representation (range) to express the array
 * data.
 */
public class ArrayPrivatization extends AnalysisPass
{
	// Map: inner loop => DEF set
	private Map def_map;

	// Map: inner loop => USE set
	private Map use_map;

	// Map: loop => private variables 
	private Map pri_map;

	// Map: loop => may private in outer loop
	private Map may_map;

	// Range map for the current procedure
	private Map range_map;

	// CFGraph for live variable analysis
	private CFGraph live_cfg;

	// Current loop
	private Statement current_loop;

	// Loop-variant variables for the current loop
	private Set loop_variants;

	// Loop-exit range domain for the current loop
	private RangeDomain exit_range;

	// Loop-exit induction increments for the current loop
	private HashMap exit_ind;

	// Whole program's alias information.
	private AliasAnalysis alias_analysis;

	// Alias result.
	private int alias_result = 0;

	// Debug
	private int debug;


	/**
	 * Constructs an array privatization analyzer for a program.
	 *
	 * @param program the input program.
	 */
	public ArrayPrivatization(Program program)
	{
		super(program);

		debug = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();

		// Call alias analysis.
		alias_analysis = new AliasAnalysis(program);
		alias_analysis.start();
	}


	/**
	 * Returns the pass name.
	 *
	 * @return the pass name in string.
	 */
	public String getPassName()
	{
		return "[Array Privatization]";
	}


	/**
	 * Starts array privatization by calling the procedure-level driver for each
	 * procedure within the program.
	 */
	public void start()
	{
		DepthFirstIterator iter = new DepthFirstIterator(program);
		iter.pruneOn(Procedure.class);

		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Procedure )
			{
				if ( alias_result != 0 )
				{
					Tools.printlnStatus(
						"[WARNING] Privatization stops due to all-to-all alias", 0);
					break;
				}
				analyzeProcedure((Procedure)o);
			}
		}
	}


	/**
	 * Starts array privatization for a procedure. It first computes the range
	 * map of the procedure to enable subsequent symbolic operations, then
	 * calls {@link #analyzeLoop(Loop)} to analyze the outer-most loops in the
	 * procedure. The final step is to annotate the analysis result for each
	 * loop in the procedure.
	 *
	 * @param proc the input procedure.
	 */
	public void analyzeProcedure(Procedure proc)
	{
		double timer = Tools.getTime();
		Tools.printlnStatus("Privatizing procedure \""+proc.getName()+"\" ...", 1);

		def_map = new HashMap();
		use_map = new HashMap();
		pri_map = new HashMap();
		may_map = new HashMap();
		range_map = (new RangeAnalysis(program)).getRangeMap(proc);

		DepthFirstIterator iter = new DepthFirstIterator(proc);
		iter.pruneOn(Loop.class);
		iter.pruneOn(StatementExpression.class); // We do not support this.

		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Loop )
				analyzeLoop((Loop)o);
		}

		addAnnotation(proc);

		Tools.printlnStatus(
			String.format("... %.2f seconds",Tools.getTime(timer)), 1);
	}


	/**
	 * Creates a cetus private annotation for each loop and inserts it before
	 * the loop.
	 */
	private void addAnnotation(Procedure proc)
	{
		// Union private variables for all loops while checking the alias
		// information
		Set pri_set = new HashSet();
		for ( Object loop : pri_map.keySet() )
		{
			Set loop_pri_set = (Set)pri_map.get(loop);
			if ( loop_pri_set == null )
				continue;
			// Use the result of alias analysis
			Set keys = new HashSet(loop_pri_set);
			for ( Object var : keys )
			{
				Set alias_set =
					alias_analysis.get_alias_set((Statement)loop, (Symbol)var);

				if ( alias_set == null )
					continue;

				for ( Object aliased : alias_set )
				{
					if ( aliased.equals("*") )
					{
						alias_result = 1;
						return; // all aliased; stop further analysis.
					}
					else if ( var != aliased )
					{
						loop_pri_set.remove(var);
						Tools.printlnStatus("  Removing aliased variable ("+
							var+","+aliased+")", 1);
						break;  // this variable has a non-empty alias set.
					}
				}
			}
			pri_set.addAll(loop_pri_set);
		}

		// Get live variables
		CFGraph live_cfg = getLiveVariables(proc, pri_set);

		for ( Object o : pri_map.keySet() )
		{
			Statement loop = (Statement)o;

			Set loop_pri_set = (Set)pri_map.get(loop);

			// Do not add annotation for non-for-loops.
			if ( loop_pri_set == null || loop_pri_set.isEmpty() )
				continue;

			DFANode loop_exit = live_cfg.getNode("stmt-exit", loop);

			HashSet loop_live_set = (HashSet)loop_exit.getData("live-out");
			if ( loop_live_set == null ) // Node that is not reachable backwards
				loop_live_set = new HashSet();
			else
				loop_live_set = (HashSet)loop_live_set.clone();

			loop_live_set.retainAll(loop_pri_set);
			loop_pri_set.removeAll(loop_live_set);
			loop_live_set.removeAll(loop_pri_set);

			// Move global variables from private => lastprivate.
			// The live variable analysis is not interprocedural.
			for ( Symbol var : new HashSet<Symbol>(loop_pri_set) )
			{
				if ( Tools.isGlobal(var, loop) )
				{
					loop_pri_set.remove(var);
					loop_live_set.add(var);
				}
			}

			Annotation note = new Annotation("cetus");
			note.setPrintMethod(Annotation.print_as_pragma_method);
			if ( !loop_pri_set.isEmpty() )
				note.add("private", loop_pri_set);
			if ( !loop_live_set.isEmpty() )
				note.add("lastprivate", loop_live_set);
			AnnotationStatement notestmt = new AnnotationStatement(note);
			notestmt.attachStatement(loop);
			CompoundStatement parent = (CompoundStatement)loop.getParent();
			parent.addStatementBefore(loop, notestmt);
		}
	}


	/**
	 * Analyzes a loop.
	 */
	private void analyzeLoop(Loop loop)
	{
		// -------------------------
		// 1. Analyze the inner loop
		// -------------------------
		DepthFirstIterator iter = new DepthFirstIterator(loop.getBody());
		iter.pruneOn(Loop.class);
		iter.pruneOn(StatementExpression.class); // We do not support this.

		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Loop )
				analyzeLoop((Loop)o);
		}

		// Prepare for the analysis; fetch environments, initialization.
		current_loop = (Statement)loop;
		loop_variants = Tools.getDefSymbol(current_loop);
		pri_map.put(loop, new HashSet());

		Tools.printlnStatus("\nLoop:\n"+current_loop, 2);

		// ------------------------------
		// 2. Build CFG for the loop body
		// ------------------------------
		CFGraph g = buildLoopGraph();

		// -----------------------------------------
		// 3. Solve reaching definition at each node
		// -----------------------------------------
		computeReachingDef(g);
		
		// Detect loop inductions; it is used when aggregating defined sections.
		computeLoopInduction(g);

		// ----------------------------
		// 4. Compute DEF, UEU, and PRI
		// ----------------------------
		// Summarize the DEF set of the loop
		def_map.put(loop, getDefSet(g));

		// Summarize the USE set of the loop
		use_map.put(loop, getUseSet(g));

		// Add USE from initialized declarations
		addUseFromDeclaration();

		// Collect private variables
		collectPrivateSet(g);

		// Aggregate DEF set and USE set.
		aggregateDef();
		aggregateUse();
	}


	/**
	 * Builds a CFG for the loop body; first it includes all control parts such as
	 * loop conditions and increments, then remove cycles to adjust the graph for
	 * the loop body analysis. It could be easier to create the CFG of the
	 * compound statement (loop body) but we also need to visit the nodes that
	 * control the loop for more general analysis of the loop.
	 */
	private CFGraph buildLoopGraph()
	{
		CFGraph ret = new CFGraph(current_loop, Loop.class);

		if ( current_loop instanceof DoLoop )
		{
			DFANode do_node = ret.getNode("stmt", current_loop);
			DFANode condition_node = (DFANode)do_node.getData("do-condition");
			DFANode back_to_node = (DFANode)condition_node.getData("true");

			back_to_node.putData("header", current_loop);
			ret.removeEdge(condition_node, back_to_node);
			ret.removeNode((DFANode)do_node.getData("do-exit"));
		}
		else if ( current_loop instanceof WhileLoop )
		{
			DFANode while_node = ret.getNode("stmt", current_loop);
			DFANode condition_node = (DFANode)while_node.getData("while-condition");

			condition_node.putData("header", current_loop);
			Set<DFANode> back_from_nodes = new HashSet(condition_node.getPreds());
			for ( DFANode back_from_node : back_from_nodes )
				if ( back_from_node.getData("stmt") != current_loop )
					ret.removeEdge(back_from_node, condition_node);
			ret.removeNode((DFANode)while_node.getData("while-exit"));
		}
		else if ( current_loop instanceof ForLoop )
		{
			DFANode for_node = ret.getNode("stmt", current_loop);
			DFANode step_node = (DFANode)for_node.getData("for-step");
			DFANode condition_node = (DFANode)for_node.getData("for-condition");

			condition_node.putData("header", current_loop);
			ret.removeEdge(step_node, condition_node);
			ret.removeNode((DFANode)for_node.getData("for-exit"));
		}

		// Remove any unreachable subgraphs.
		ret.topologicalSort(ret.getNodeWith("stmt","ENTRY"));
		List<DFANode> removable = new ArrayList<DFANode>();
		Iterator<DFANode> iter = ret.iterator();
		while ( iter.hasNext() )
		{
			DFANode curr = iter.next();
			if ( (Integer)curr.getData("top-order") < 0 )
				removable.add(curr);
		}
		for ( DFANode curr : removable )
			ret.removeNode(curr);

		// Add RangeDomain to each nodes.
		getRangeDomain(ret);

		Tools.printlnStatus(ret.toDot("range,ir,tag", 2), 3);

		return ret;
	}


	/**
	 * Collects USE from the initialized declarations.
	 */
	private void addUseFromDeclaration()
	{
		Section.MAP new_use_map = (Section.MAP)use_map.get(current_loop);
		RangeDomain rd = (RangeDomain)range_map.get(current_loop);
		if ( rd == null )
			rd = new RangeDomain();
		else
			rd = (RangeDomain)rd.clone();
		DepthFirstIterator iter =
			new DepthFirstIterator(((Loop)current_loop).getBody());
		iter.pruneOn(Loop.class);

		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof Initializer )
				for ( Object child : ((Initializer)o).getChildren() )
					if ( child instanceof Traversable )
						for ( Expression use : Tools.getUseSet((Traversable)child) )
							new_use_map = new_use_map.unionWith(getSectionMap(use,false), rd);
		}

		use_map.put(current_loop, new_use_map);
	}


	/**
	 * Computes the reaching definition for each node in the CFG.
	 */
	private void computeReachingDef(CFGraph g)
	{
		TreeMap work_list = new TreeMap();

		// Enter the entry node in the worklist.
		DFANode entry = g.getNodeWith("stmt", "ENTRY");
		entry.putData("def-in", new Section.MAP());
		work_list.put(entry.getData("top-order"), entry);

		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.firstKey());

			Section.MAP curr_map = null;

			for ( DFANode pred : node.getPreds() )
			{
				Section.MAP pred_map = (Section.MAP)pred.getData("def-out");

				if ( curr_map == null )
					curr_map = (Section.MAP)pred_map.clone();
				else
					curr_map = curr_map.intersectWith(pred_map,
						(RangeDomain)node.getData("range"));
			}

			Section.MAP prev_map = (Section.MAP)node.getData("def-in");

			if ( prev_map == null || !prev_map.equals(curr_map) )
			{
				node.putData("def-in", curr_map);

				// Handles data kill, union, etc.
				computeOutDef(node);

				for ( DFANode succ : node.getSuccs() )
					work_list.put(succ.getData("top-order"), succ);
			}
		}
	}


	/**
	 * Computes DEFout from DEFin for the node while collecting candidate private
	 * variables; if a defined variable is a scalar or an array with
	 * loop-invariant subscript expression, it is a candidate.
	 * If the node contains any function calls, the following conservative
	 * decision is made:
	 *   Every global variables and actual parameters of which addresses are
	 *   taken are considered as being modified.
	 *
	 * This decision affects the analysis in two aspects:
	 * 1) Array section containing the above variables should be killed.
	 * 2) Used array sections containing the above variables are overly
	 *    approximated and we handle this case by removing the DEF array section
	 *    entries for the variables
	 */
	private void computeOutDef(DFANode node)
	{
		Section.MAP in = new Section.MAP(), out = null;
		RangeDomain rd = (RangeDomain)node.getData("range");
		Set<Symbol> killed_vars = new HashSet<Symbol>();
		Set<Symbol> pri_set = (Set<Symbol>)pri_map.get(current_loop);

		if ( node.getData("def-in") != null )
			in = (Section.MAP)((Section.MAP)node.getData("def-in")).clone();

		if ( node.getData("super-entry") != null  )
		{
			Statement loop = (Statement)node.getData("super-entry");

			out = (Section.MAP)def_map.get(loop);

			killed_vars.addAll(Tools.getDefSymbol(loop));

			// Add may private variables in the candidate list. These variables are
			// collected while aggregating the inner loops. See aggregateDef() for
			// more details.
			Section.MAP may_set = (Section.MAP)may_map.get(loop);
			pri_set.addAll(may_set.keySet());
		}
		else
		{
			out = new Section.MAP();

			Object o = CFGraph.getIR(node);

			if ( o instanceof Traversable )
			{
				Traversable tr = (Traversable)o;

				for ( Expression e : Tools.getDefSet(tr) )
					out = out.unionWith(getSectionMap(e, true), rd);

				killed_vars.addAll(Tools.getDefSymbol(tr));

				// Kill DEF section containing globals and actual parameters.
				in.removeSideAffected(tr);
			}
		}

		// Kill DEF section containing killed variables.
		in.removeAffected(killed_vars);

		// Candidate collection
		setPrivateCandidates(out);

		Section.MAP unioned = in.unionWith(out, rd);

		node.putData("def-out", unioned);
	}


	/**
	 * Solves additive progressions for each node; this information is used when
	 * aggregating MUST sections; the following equation describes this problem.
	 *
	 * IN[n]  = (V) { OUT[p], p in predecessors of n }
	 * OUT[n] = IN[n] (^) GEN[n]
	 * (v): join like constant propagation
	 * (^): add induction increments
	 *
	 * Each variables' states are initialized to 0.
	 * Non-induction assignment just kills the data.
	 * Example with OUT data:
	 * i=i+1 : i=>1
	 * i=i+2 : i=>3
	 * i=k   : i=>TOP
	 */
	private void computeLoopInduction(CFGraph g)
	{
		TreeMap work_list = new TreeMap();

		// Enter the entry node in the worklist.
		DFANode entry = g.getNodeWith("stmt", "ENTRY");
		work_list.put(entry.getData("top-order"), entry);

		boolean in_loop_body = false;

		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.firstKey());

			HashMap curr_map = null;

			for ( DFANode pred : node.getPreds() )
			{
				HashMap pred_map = (HashMap)pred.getData("ind-out");

				// Unreachable nodes do not have "ind-out".
				if ( pred_map == null )
					pred_map = new HashMap();

				if ( curr_map == null )
					curr_map = (HashMap)pred_map.clone();
				else
					curr_map = joinInductionMap(curr_map, pred_map);
			}

			if ( curr_map == null )
				curr_map = new HashMap();

			HashMap prev_map = (HashMap)node.getData("ind-in");

			if ( prev_map == null || !prev_map.equals(curr_map) )
			{
				node.putData("ind-in", curr_map);

				if ( in_loop_body )
					meetInductionMap(node);
				else
				{
					node.putData("ind-out", curr_map);
					if ( node.getData("header") != null )
						in_loop_body = true;
				}

				for ( DFANode succ : node.getSuccs() )
					work_list.put(succ.getData("top-order"), succ);
			}
		}
	}


	/**
	 * Returns the result of a join operation of the two induction map.
	 *  e1 v e2 = e1  if e1==e2,
	 *          = T   otherwise 
	 */
	private HashMap joinInductionMap(HashMap m1, HashMap m2)
	{
		HashMap ret = new HashMap();

		for ( Object var : m1.keySet() )
		{
			Object expr = m2.get(var);
			if ( expr != null && expr.equals(m1.get(var)) )
				ret.put(var, expr);
			else
				ret.put(var, "T");
		}

		for ( Object var : m2.keySet() )
			ret.put(var, "T");

		return ret;
	}


	/**
	 * Performs a meet operation in the specified node (transfer function).
	 *  e1 ^ e2 = e1+e2  if e1!=T and e2!=T
	 *          = T      otherwise
	 */
	private void meetInductionMap(DFANode node)
	{
		HashMap in = (HashMap)node.getData("ind-in");

		HashMap out = (HashMap)in.clone();

		Object o = CFGraph.getIR(node);

		if ( !(o instanceof Traversable) )
		{
			node.putData("ind-out", out);
			return;
		}

		DepthFirstIterator iter = new DepthFirstIterator((Traversable)o);

		while ( iter.hasNext() )
		{
			Object e = iter.next();

			Symbol var = null;
			Object diff = null;

			if ( e instanceof AssignmentExpression )
			{
				AssignmentExpression ae = (AssignmentExpression)e;
				if ( ae.getLHS() instanceof Identifier &&
				Tools.isInteger(((Identifier)ae.getLHS()).getSymbol()) )
				{
					var = ((Identifier)ae.getLHS()).getSymbol();
					AssignmentOperator ao = ae.getOperator();
					if ( ao == AssignmentOperator.NORMAL )
					{
						diff = NormalExpression.subtract(ae.getRHS(),ae.getLHS());
						if ( Tools.containsSymbol((Expression)diff, var) )
							diff = "T";
					}
					else if ( ao == AssignmentOperator.ADD )
						diff = (Expression)ae.getRHS().clone();
					else if ( ao == AssignmentOperator.SUBTRACT )
						diff = NormalExpression.subtract(new IntegerLiteral(0),ae.getRHS());
					else
						diff = "T";
				}
			}
			else if ( e instanceof UnaryExpression )
			{
				UnaryExpression ue = (UnaryExpression)e;
				if ( ue.getExpression() instanceof Identifier &&
				Tools.isInteger(((Identifier)ue.getExpression()).getSymbol()) )
				{
					var = ((Identifier)ue.getExpression()).getSymbol();
					UnaryOperator uo = ue.getOperator();
					if ( uo.toString().equals("++") )
						diff = new IntegerLiteral(1);
					else if ( uo.toString().equals("--") )
						diff = new IntegerLiteral(-1);
					else
						var = null;
				}
			}

			if ( var != null )
			{
				if ( Tools.containsClass((Traversable)o, ConditionalExpression.class) )
					diff = "T"; // Conservatively assume this is a may diff
				Object prev_diff = out.get(var);
				if ( prev_diff == null )
					out.put(var, diff);
				else if ( prev_diff.equals("T") || diff.equals("T") )
					out.put(var, "T");
				else if ( (prev_diff instanceof Expression) &&
				(diff instanceof Expression) )
					out.put(var, NormalExpression.add(
						(Expression)prev_diff, (Expression)diff));
				else
					out.put(var, "T");
			}
		}

		node.putData("ind-out", out);
	}



	/**
	 * Examines each function call and collect arrays that may be used in the
	 * function's callee. Scalars are not considered in this method because
	 * whether they (scalars) are used is not important for the analysis;
	 * remember that this is for conservative array section operations in the
	 * absence of any interprocedural analysis.
	 */
	private Set<Symbol> getMayUseFromFC(Traversable tr, Set<Symbol> vars)
	{
		Set<Symbol> ret = new HashSet<Symbol>();
		Set<Symbol> local_array = new HashSet<Symbol>();

		for ( Symbol var : vars )
		{
			if ( Tools.isArray(var) )
			{
				if ( Tools.isGlobal(var, tr) )
					ret.add(var);
				else
					local_array.add(var);
			}
		}

		DepthFirstIterator iter = new DepthFirstIterator(tr);
		iter.pruneOn(FunctionCall.class);

		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( o instanceof FunctionCall )
			{
				for ( Symbol var : local_array )
				{
					if ( Tools.containsSymbol((Traversable)o, var) )
						ret.add(var);
				}
			}
		}

		return ret;
	}


	/**
	 * Collects candidate private variables from the section map by checking if
	 * they do not contain any loop-variant variables.
	 */
	private void setPrivateCandidates(Section.MAP m)
	{
		Set<Symbol> pri_set = (Set<Symbol>)pri_map.get(current_loop);

		for ( Symbol var : m.keySet() )
		{
			Section section = m.get(var);

			if ( section.isScalar() )
				pri_set.add(var);
			else if ( !section.containsSymbols(loop_variants) )
				pri_set.add(var);
			else if ( pri_set.contains(var) )
				pri_set.remove(var);
		}
	}


	/**
	 * Computes the DEF summary set.
	 */
	private Map getDefSet(CFGraph g)
	{
		Section.MAP ret = null;
		exit_range = null;
		exit_ind = null;
		// exit_range is a common range domain (union) for the exiting nodes.

		for ( int i=0; i < g.size(); ++i )
		{
			DFANode node = g.getNode(i);

			// Skip unreachable nodes (see HSG generation in CFGraph)
			Integer top_order = (Integer)node.getData("top-order");
			if ( top_order == null || top_order < 0 )
				continue;

			if ( node.getSuccs().size() > 0 )
				continue;

			RangeDomain rd = (RangeDomain)node.getData("range");

			HashMap curr_ind = (HashMap)node.getData("ind-out");

			// Node with no successors is outside of the loop body, so let's
			// take def-in instead of def-out to avoid data kills due to the
			// loop step increments.
			Section.MAP curr_map = (Section.MAP)node.getData("def-in");

			if ( ret == null )
			{
				exit_range = (RangeDomain)rd.clone();
				exit_ind = curr_ind;
				ret = (Section.MAP)curr_map.clone();
			}
			else
			{
				exit_range.unionRanges(rd);
				exit_ind = joinInductionMap(exit_ind, curr_ind);
				ret = ret.intersectWith(curr_map, exit_range);
			}
		}

		ret.clean();

		return ret;
	}


	/**
	 * Computes the UEU summary set; UEU = UEU + (USE-DEF)
	 */
	private Map getUseSet(CFGraph cfg)
	{
		Section.MAP ret = new Section.MAP();

		Iterator<DFANode> cfgiter = cfg.iterator();
		while ( cfgiter.hasNext() )
		{
			DFANode node = cfgiter.next();

			// Skip unreachable nodes (see HSG generation in CFGraph)
			Integer top_order = (Integer)node.getData("top-order");
			if ( top_order == null || top_order < 0 )
				continue;

			RangeDomain rd = (RangeDomain)node.getData("range");

			Section.MAP local_use = new Section.MAP();

			// Super node
			if ( node.getData("super-entry") != null )
				local_use = (Section.MAP)use_map.get(node.getData("super-entry"));

			// Other node
			else
			{
				Object o = CFGraph.getIR(node);

				if ( !(o instanceof Traversable) )
					continue;
				
				for ( Expression e : Tools.getUseSet((Traversable)o) )
					local_use = local_use.unionWith(getSectionMap(e, false), rd);
			}

			Section.MAP in_def = (Section.MAP)node.getData("def-in");

			Section.MAP local_ueu = local_use.differenceFrom(in_def, rd);

			ret = ret.unionWith(local_ueu, rd);
		}
		
		return ret;
	}


	/**
	 * Collects privatizable variables. The candidate private variables have
	 * already been stored in the pri_map for the analyzer. If the summary UEU
	 * set of each loop contains any candidate variable, that variable is removed
	 * from the candidate list. Another set of candidates come from the may_set
	 * which contains the defined set of the inner loops with the execution of
	 * the loop not guaranteed at compile time. This set needs to be added to the
	 * private list after excluding UEU set.
	 */
	private void collectPrivateSet(CFGraph g)
	{
		Set pri_set = (Set)pri_map.get(current_loop);

		Section.MAP may_set = new Section.MAP();

		// Do not collect for certain loops that are not parallelizable.
		// While loops, Do loops and loops containing conditional exits.
		// Other information such as DEF/USE set is still useful for those loops
		// for the analysis of outer loops.
		if ( current_loop instanceof ForLoop )
		{
			Iterator<DFANode> node_iter = g.iterator();
			while ( node_iter.hasNext() )
			{
				DFANode node = node_iter.next();
				Object ir = CFGraph.getIR(node);

				// 1. GotoStatement getting out of the loop.
				// 2. BreakStatement that breaks the current loop.
				// 3. ReturnStatement.
				if ( ir instanceof GotoStatement &&
				g.getNode("ir", ((GotoStatement)ir).getTarget()) == null ||
				ir instanceof BreakStatement && node.getSuccs().isEmpty() ||
				ir instanceof ReturnStatement )
				{
					pri_set.clear();
					continue;
				}

				// Collect delayed private variables.
				Statement loop = (Statement)node.getData("super-entry");
				if ( loop != null )
				{
					Section.MAP loop_may_set = (Section.MAP)may_map.get(loop);
					may_set = may_set.unionWith(loop_may_set, exit_range);
				}
			}
		}
		else
			pri_set.clear();

		Set use_set = ((Map)use_map.get(current_loop)).keySet();

		Set def_set = ((Map)def_map.get(current_loop)).keySet();

		// Locally declared variables
		Set local_set =
			Tools.getVariableSymbols((SymbolTable)((Loop)current_loop).getBody());

		// Remove local_set from the DEF/USE set
		def_set.removeAll(local_set);
		use_set.removeAll(local_set);

		// Collect private variables
		pri_set.removeAll(use_set);
		pri_set.removeAll(local_set);

		// Variables may be defined in inner loops need to be tested if there is
		// any read access to these variables. If there is no use point for those
		// variables, they are still considered to be private and marked as being
		// defined to be considered as private candidate in the outer loop.
		may_set.keySet().removeAll(use_set);
		pri_set.addAll(may_set.keySet());
		Section.MAP recovered_def = (Section.MAP)def_map.get(current_loop);
		def_map.put(current_loop, recovered_def.unionWith(may_set, exit_range));

		// Conservatively remove variables from the private set if they appear
		// in function call parameters or they are global; will be improved.
		removeMayUsedVariables(pri_set);
		removeMayUsedVariables(def_set);

		// Remove any user-declared type; will be improved.
		removeUserTypes(pri_set);
		removeUserTypes(def_set);

		Tools.printlnStatus("DEF = "+def_map.get(current_loop), 2);
		Tools.printlnStatus("USE = "+use_map.get(current_loop), 2);
		Tools.printlnStatus("PRI = "+pri_map.get(current_loop), 2);
	}


	/**
	 * Removes any variables typed with user-defined types.
	 */
	private void removeUserTypes(Set vars)
	{
		Set keys = new HashSet(vars);
		for ( Object var : keys )
		{
			if ( var == null )
			{
				vars.remove(var);
				continue;
			}
			List types = ((Symbol)var).getTypeSpecifiers();
			if ( types != null )
			{
				for ( Object type : types )
				{
					if ( type instanceof UserSpecifier )
					{
						vars.remove(var);
						break;
					}
				}
			}
		}
	}


	/**
	 * Removes variables from the set if they appear in function call or they
	 * are global when a function call exists.
	 */
	private void removeMayUsedVariables(Set vars)
	{
		DepthFirstIterator iter = new DepthFirstIterator(current_loop);
		FunctionCall fc = null;
		while ( iter.hasNext() )
		{
			Object o = iter.next();
			if ( !(o instanceof FunctionCall) )
				continue;

			fc = (FunctionCall)o;

			// Referencing
			List<UnaryExpression> address_of =
				Tools.getUnaryExpression(fc, UnaryOperator.ADDRESS_OF);
			for ( UnaryExpression ue : address_of )
				vars.remove(Tools.getSymbolOf(ue.getExpression()));

			// Pointer type and user type
			Set<Symbol> params = Tools.getAccessedSymbols(fc);
			//vars.removeAll(params) ; conservative
			for ( Symbol param : params )
			{
				if ( vars.contains(param) )
				{
					List spec = param.getTypeSpecifiers();
					if ( Tools.containsClass(spec, PointerSpecifier.class) ||
					Tools.containsClass(spec, UserSpecifier.class) )
						vars.remove(param);
				}
			}
		}

		// Global variables.
		if ( fc != null )
		{
			Set keys = new HashSet(vars);
			for ( Object var : keys )
			{
				if ( Tools.isGlobal((Symbol)var, current_loop) )
					vars.remove(var);
			}
		}
	}


	/**
	 * Computes set of live variables at each program point with the given
	 * mask_set being the universal set.
	 */
	private CFGraph getLiveVariables(Traversable t, Set mask_set)
	{
		CFGraph g = new CFGraph(t);

		g.topologicalSort(g.getNodeWith("stmt", "ENTRY"));

		TreeMap work_list = new TreeMap();

		List<DFANode> exit_nodes = g.getExitNodes();

		for ( DFANode exit_node : exit_nodes )
			work_list.put(exit_node.getData("top-order"), exit_node);

		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.lastKey());

			// LIVEout
			HashSet live_out = new HashSet();
			for ( DFANode succ : node.getSuccs() )
			{
				Set succ_in = (Set)succ.getData("live-in");
				if ( succ_in != null )
					live_out.addAll(succ_in);
			}

			// Convergence
			Set prev_live_out = (Set)node.getData("live-out");
			if ( prev_live_out != null && prev_live_out.equals(live_out) )
				continue;

			node.putData("live-out", live_out.clone());

			// Local computation
			Set gen = new HashSet(), kill = new HashSet();
			computeLiveVariables(node, gen, kill, mask_set);

			// LiveIn = Gen (v) ( LiveOut - Kill )
			live_out.removeAll(kill);
			live_out.addAll(gen);

			// Intersect with the masking set (reduces the size of the live set)
			live_out.retainAll(mask_set);

			node.putData("live-in", live_out);

			for ( DFANode pred : node.getPreds() )
				work_list.put(pred.getData("top-order"), pred);
		}

		return g;
	}


	/**
	 * Transfer function for live variable analysis.
	 */
	private void computeLiveVariables
	(DFANode node, Set gen, Set kill, Set mask_set)
	{
		Object tr = CFGraph.getIR(node);

		if ( !(tr instanceof Traversable) ) // symbol-entry with initializer?
			return;

		gen.addAll(Tools.getUseSymbol((Traversable)tr));

		kill.addAll(Tools.getDefSymbol((Traversable)tr));

		// Conservative decision on funcion calls; add any variables in the
		// mask_set to the GEN set.
		if ( Tools.containsClass((Traversable)tr, FunctionCall.class) )
		{
			for ( Object var : mask_set )
				if ( Tools.isGlobal((Symbol)var, current_loop) )
					gen.add(var);
		}

		// Name only support for access expressions.
		addAccessName(gen);
		addAccessName(kill);

		return;
	}


	/**
	 * Adds the base name of an access expression to the given set to enable
	 * name-only analysis.
	 */
	private void addAccessName(Set set)
	{
		Set vars = new HashSet(set);
		for ( Object var : vars )
			if ( var instanceof AccessSymbol )
				set.add(((AccessSymbol)var).get(0));
	}


	/**
	 * Aggregates MUST DEF set of the loop.
	 */
	private void aggregateDef()
	{
		// Additional information about zero-trip loop consideration.
		RangeDomain entry_range = (RangeDomain)range_map.get(current_loop);
		if ( entry_range == null )
			entry_range = new RangeDomain();
		else
			entry_range = (RangeDomain)entry_range.clone();
		int eval = 1; // eval == 1 means the loop is executed at least once.
		Set init_set = new HashSet();

		if ( current_loop instanceof ForLoop )
		{
			ForLoop for_loop = (ForLoop)current_loop;
			// Add range from the initial statement.
			entry_range.intersectRanges(
				RangeAnalysis.getRangeDomain(for_loop.getInitialStatement()));
			eval=entry_range.evaluateLogic(for_loop.getCondition());
			init_set.addAll(Tools.getDefSymbol(for_loop.getInitialStatement()));
		}
		else if ( current_loop instanceof WhileLoop )
		{
			WhileLoop while_loop = (WhileLoop)current_loop;
			eval=entry_range.evaluateLogic(while_loop.getCondition());
		}

		Section.MAP may_set = new Section.MAP();

		Section.MAP defs = (Section.MAP)def_map.get(current_loop);
		Set<Symbol> vars = new HashSet<Symbol>(defs.keySet());

		for ( Symbol var : vars )
		{
			Set ivs = getInductionVariable();

			Section before = defs.get(var);
			Section section = (Section)before.clone();
			section.expandMust(exit_range, ivs, loop_variants);

			//Tools.printlnStatus("DEF=>DEFS:"+var+" = "+before+"=>"+section, 2);

			if ( section.isArray() && section.isEmpty() )
			{
				defs.remove(var);
				continue;
			}

			// If the loop is not executed in any case, move the sections to the
			// may_set and the outer loop makes decision whether the elements of the
			// may_set is private.
			if ( section.equals(before) )
			{
				if ( eval != 1 && !init_set.contains(var) )
				{
					may_set.put(var, section);
					defs.remove(var);
				}
			}
			else
				defs.put(var, section);
		}
		
		may_map.put(current_loop, may_set);

		Tools.printlnStatus("DEFS = "+defs, 2);
	}


	/**
	 * Aggregates MAY USE set of the loop.
	 */
	private void aggregateUse()
	{
		Section.MAP my_use = (Section.MAP)use_map.get(current_loop);

		Set<Symbol> vars = new HashSet<Symbol>(my_use.keySet());

		for ( Symbol var : vars )
			my_use.get(var).expandMay(exit_range, loop_variants);

		Tools.printlnStatus("USES = "+my_use, 2);
	}


	/**
	 * Returns a set of induction variables which are used in the aggregation
	 * process in the future.
	 */
	private Set getInductionVariable()
	{
		Set ret = new HashSet();

		for ( Object var : exit_ind.keySet() )
		{
			Object stride = exit_ind.get(var);
			if ( stride instanceof Expression &&
			((Expression)stride).equals(new IntegerLiteral(1)) )
				ret.add(var);
		}

		return ret;
	}


	/**
	 * Returns map from a variable to its section
	 */
	private Section.MAP getSectionMap(Expression e, boolean def)
	{
		Section.MAP ret = new Section.MAP();

		if ( e instanceof ArrayAccess )
		{
			ArrayAccess aa = (ArrayAccess)e;

			Symbol var = Tools.getSymbolOf(aa.getArrayName());

			if ( var instanceof AccessSymbol )
			{
				// Only use set is considered important for name-only analysis because
				// it implies more conservative analysis.
				if ( !def )
					ret.put( ((AccessSymbol)var).get(0), new Section(-1) );
			}
			else
				ret.put(Tools.getSymbolOf(aa.getArrayName()), new Section(aa));
		}
		else if ( e instanceof AccessExpression )
		{
			if ( !def )
			{
				Set use_set = Tools.getUseSet(e);
				if ( use_set.size() == 1 )
				{
					AccessSymbol var = (AccessSymbol)Tools.getSymbolOf(e);
					ret.put( var.get(0), new Section(-1) );
				}
			}
		}
		else
		{
			Symbol var = Tools.getSymbolOf(e);
			// var == null means it is not variable type
			// e.g.) *a = 0;
			if ( var != null )
				ret.put(var, new Section(-1));
		}

		ret.clean();

		return ret;
	}


	/**
	 * For now, just union them once in reverse post order.
	 * This method is inefficient because we need to rebuild RangeDomain for
	 * intermediate nodes which do not represent a statement in the IR.
	 * It seems like this is the only way to provide range information correctly.
	 */
	private void getRangeDomain(CFGraph g)
	{
		TreeMap<Integer,DFANode> work_list = new TreeMap<Integer,DFANode>();

		Iterator<DFANode> iter = g.iterator();
		while ( iter.hasNext() )
		{
			DFANode node = iter.next();
			work_list.put((Integer)node.getData("top-order"), node);
		}

		for ( Integer order : work_list.keySet() )
		{
			DFANode node = work_list.get(order);

			Object ir = node.getData(Arrays.asList("super-entry","stmt"));

			RangeDomain rd = (RangeDomain)range_map.get(ir);

			if ( rd != null )
			{
				node.putData("range", rd.clone());
			}
			else if ( order == 0 )
			{
				RangeDomain range = (RangeDomain)range_map.get(current_loop);

				if ( range == null )
					node.putData("range", new RangeDomain());
				else
					node.putData("range", range.clone());
			}
			else
			{
				RangeDomain range = null;
				for ( DFANode pred : node.getPreds() )
				{
					RangeDomain pred_range = (RangeDomain)pred.getData("range");

					if ( pred_range == null )
						pred_range = new RangeDomain();

					if ( range == null )
						range = (RangeDomain)pred_range.clone();
					else
						range.unionRanges(pred_range);
				}
				node.putData("range", range);
			}
		}
	}

}
