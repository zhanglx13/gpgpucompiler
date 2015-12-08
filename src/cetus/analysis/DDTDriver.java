package cetus.analysis;

import java.util.*;

import cetus.hir.*;

/*
* Algorithm:
* ---- Traverse IR ----
* Traverse across outermost loops
* analyze(loop) checks every outermost loop and inner nests for eligibility
* 
* Eligibility:
* 	- Canonical loops (Fortran type DO loop format) - See LoopTools.java for details
* 	- Perfect Nest
* 	- Cannot contain function calls
* 	- No control flow modifiers within loop
*
* ----- Collect Info -----
* if(eligible)
* 	collect loop information
* 		- loop bounds, increments, index variable
* 		- array accesses in entire nest
* 
* ----- Run test ----
* Test for data dependence between accesses of the same array within the entire
* loop nest, obtain direction vector set and build a Data Dependence Graph (DDG)
* 	- Check for privatization, reduction and alias analysis before dependence testing
*
*/

/**
* Performs array data-dependence analysis on eligible loop nests in the program
*/
public class DDTDriver extends AnalysisPass
{
	/* Map from outermost loop in the nest to the dependence graph for that nest */
	private HashMap<Loop, DDGraph> ddGraphMap;

	/**
	 * Constructor
	 */
	public DDTDriver(Program program)
	{
		super(program);
	}

	/** 
	 * Performs whole program data-dependence analysis 
	 */
	public void start()
	{
		ddGraphMap = new HashMap<Loop, DDGraph>();
			
		/* Iterate breadth-first over outermost loops in the program */
		BreadthFirstIterator bfs_iter = new BreadthFirstIterator(program);
		bfs_iter.pruneOn(Loop.class);
		for (;;)
		{
			Loop loop = null;
			try {
				loop = (Loop)bfs_iter.next(Loop.class);
			} catch (NoSuchElementException e) {
				break;
			}
			
			/* Analyze this loop and it's inner nest for dependence */
			DDGraph dependence_graph = analyzeLoop(loop);
			System.out.println("analyze: \n"+loop);
			if (dependence_graph != null)
				ddGraphMap.put(loop, dependence_graph);
		}
	}

	public String getPassName()
	{
		return new String("[DDT]");
	}

	/**
	 * Analyze this loop and all inner loops to build a dependence graph for the loop nest
	 * with this loop as the outermost loop
	 */
	public DDGraph analyzeLoop(Loop loop)
	{
		/* Dependence Graph for nest defined by this loop */
		DDGraph loopDDGraph = new DDGraph();
		/* Storage space for loop information */
		HashMap <Loop, LoopInfo>loopInfoMap = 
			new HashMap<Loop, LoopInfo>();
		/* Map from array name (Expression) to array access information (DDArrayAccessInfo) */
		HashMap <Symbol, ArrayList<DDArrayAccessInfo>>loopArrayAccessMap = 
			new HashMap<Symbol, ArrayList<DDArrayAccessInfo>>();

		/* Collect loop information and array access information */
		boolean eligible = collectLoopInfo(loop, loopInfoMap, loopArrayAccessMap);
		if (eligible == false)
		{
			Tools.println("One or more loops in the nest are not eligible for data-dependence analysis", 2);
			return null;
		}

		/* Iterate over nested inner loops */
		DepthFirstIterator dfs_iter = new DepthFirstIterator((Traversable)loop.getBody());
		for (;;)
		{
			Loop nested_loop = null;

			try {
				nested_loop = (Loop)dfs_iter.next(Loop.class);
			} catch (NoSuchElementException e) {
				break;
			}
			
			/* Collect loop information and array access information */
			eligible = collectLoopInfo(nested_loop, loopInfoMap, loopArrayAccessMap);
			if (eligible == false)
			{
				Tools.println("One or more loops in the nest are not eligible for data-dependence analysis", 2);
				return null;
			}
		}
		
		/* Run Data-Dependence Tests for entire nest and return the DDG if everything went OK */
		loopDDGraph = runDDTest(loop, loopArrayAccessMap, loopInfoMap);
		return loopDDGraph;
	}
	

	/**
	 * Returns a dependence graph map for the whole program
	 * @param summarize Specify whether direction vectors should be summarized or not
	 * @return Whole program dependence graph map
	 */
	public HashMap<Loop, DDGraph> getDDGraph(boolean summarize)
	{
		if (summarize)
		{
			Set<Loop> loopKey = ddGraphMap.keySet();

			for (Loop loop : loopKey)
			{
				(ddGraphMap.get(loop)).summarizeGraph();
			}

			return ddGraphMap;
		}
		else
			return ddGraphMap;
	}
	
	/**
	 * Performs data-dependence analysis on the nest defined by loop and returns
	 * the dependence graph
	 * @param summarize: Specify whether direction vectors should be summarized or not
	 * @param loop: Loop that defines the nest to be analyzed
	 * @return DDGraph - dependence graph for nest (null if loop nest is not 
	 * eligible for dependence analysis)
	 */
	public DDGraph getDDGraph(boolean summarize, Loop loop)
	{
		DDGraph loopDDGraph = analyzeLoop(loop);
		if (loopDDGraph != null)
		{
			if(summarize)
			{
				loopDDGraph.summarizeGraph();
				return loopDDGraph;
			}
			else
				return loopDDGraph;
		}
		// Loop nest was not eligible for dependence analysis
		else
			return null;
	}
	
	/**
	 * Analyze every loop to gather and store loop information
	 * such as upper and lower bounds, increment, index variable.
	 * Also build framework required for calling DD tests
	 * 
	 * @return boolean
	 */
	private boolean collectLoopInfo(Loop currentLoop, HashMap<Loop, LoopInfo>loopInfoMap,
			HashMap<Symbol, ArrayList<DDArrayAccessInfo>>loopArrayAccessMap)
	{
		if (isEligible(currentLoop) == false) 
		{
			if (currentLoop instanceof ForLoop)
			{ 
				Tools.println("The following loop is not eligible for DDTest: ", 2);
				Tools.println(((ForLoop)currentLoop).toString(), 2);
				Tools.println("----------------------------------------------", 2);
			}
			else
			{
				Tools.println("This loop is not eligible for DDTest (not a for loop)", 2);
			}
			return false;
		}

		/* Gather loop information and store it in a map with the loop as the key */
		LoopInfo loop_info = new LoopInfo(currentLoop);
		loopInfoMap.put(currentLoop, loop_info);

		/* get write and read array accesses only for this loop body */
		addWriteAccesses(currentLoop, loopArrayAccessMap, (Traversable)currentLoop.getBody());
		addReadAccesses(currentLoop, loopArrayAccessMap, (Traversable)currentLoop.getBody());
		return true;
	}

	/**
	 * Checks for current scope of DDT framework. The scope of the framework
	 * will be broadened in future releases.
	 */
 
	/* Checks:
	 * 1. isCanonical
	 * 		for(i = 0; i <= n; i++) --> for(initial constant assignment;
	 * 										condition on upper bound;
	 * 										constant increment)
	 * 2. containsFunctionCall
	 * 3. containsControlFlowModifier
	 * 		Do not test loops with function calls/labels/return_stmt/break_stmt/goto_stmt 
	 * 		side-effects not handled
	 * 4. isPerfectNest
	 * 		Does not currently support array dependence analysis for non-perfectly
	 * 		nested loops
	 * 5. constantLoopBounds
	 * 		Lower bound must be constant. If upper bound is symbolic,
	 * 		currently performs dependence analysis for maximum range (Long.MAX_VALUE)
	 */
	private boolean isEligible(Loop loop)
	{
		/* Checks whether the loop is in a conventional
		 * Fortran type do loop format */
		if (LoopTools.isCanonical(loop)==false)
		{
			Tools.println("Loop is not canonical", 2);
			return false;
		}
		/* One of the checks to see that the loop body contains 
		 * no side-effects */
		if (LoopTools.containsFunctionCall(loop)==true)
		{
			Tools.println("Loop can have side-effects", 2);
			return false;
		}
		/* Currently deal only with perfectly nested loops */
		if (LoopTools.isPerfectNest(loop)==false)
		{
			Tools.println("Loop does not contain a perfect nest", 2);
			return false;
		}
		/* Check if loop bounds are numeric constants */
		if (LoopTools.isLowerBoundConstant(loop)==false)
		{
			Tools.println("Loop lower bound is not integer constant", 2);
			return false;
		}
		/* Check if loop bounds are numeric constants 
		 * Currently symbolic upper bound is treated as max_int value */
		//if (LoopTools.isUpperBoundConstant(loop)==false)return false;

		/* Check if loop contains branching labels, go-to statements or break statements */
		if (LoopTools.containsControlFlowModifier(loop)==true)
		{
			Tools.println("Loop can have side-effects", 2);
			return false;
		}

		return true;
	}


	private void addWriteAccesses(Loop loop,
			HashMap<Symbol, ArrayList<DDArrayAccessInfo>>loopArrayAccessMap,
			Traversable root)
	{

		if (root instanceof Expression)
		{
			BreadthFirstIterator iter = new BreadthFirstIterator(root);

			HashSet<Class> of_interest = new HashSet<Class>();
			of_interest.add(AssignmentExpression.class);
			of_interest.add(UnaryExpression.class);

			for (;;)
			{
				Object o = null;

				try {
					o = iter.next(of_interest);
				} catch (NoSuchElementException e) {
					break;
				}

				if (o instanceof AssignmentExpression)
				{
					AssignmentExpression expr = (AssignmentExpression)o;

					/*
					 *  Only the left-hand side of an AssignmentExpression
					 *  is a definition.  There may be other nested
					 *  definitions but, since iter is not set to prune
					 *  on AssignmentExpressions, they will be found during
					 *  the rest of the traversal.
					 */
					if (expr.getLHS() instanceof ArrayAccess)
					{
						Statement stmt = (expr.getLHS()).getStatement();
						DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
								(ArrayAccess)expr.getLHS(), DDArrayAccessInfo.write_type, loop, stmt);
						addArrayAccess(arrayInfo, loopArrayAccessMap);
					}
				}
				else
				{
					UnaryExpression expr = (UnaryExpression)o;
					UnaryOperator op = expr.getOperator();

					/*
					 *  there are only a few UnaryOperators that create definitions
					 */
					if (UnaryOperator.hasSideEffects(op) &&
							expr.getExpression() instanceof ArrayAccess)
					{
						Statement stmt = (expr.getExpression()).getStatement();
						DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
								(ArrayAccess)expr.getExpression(), DDArrayAccessInfo.write_type, loop, stmt);
						addArrayAccess(arrayInfo, loopArrayAccessMap);
					}
				}
			}
		}
		else if (root instanceof IfStatement)
		{
			IfStatement if_stmt = (IfStatement)root;

			addWriteAccesses(loop, loopArrayAccessMap, if_stmt.getThenStatement());

			if (if_stmt.getElseStatement() != null)
			{
				addWriteAccesses(loop, loopArrayAccessMap, if_stmt.getElseStatement()); 
			}
		}
		else if (root instanceof Loop)
		{
		}
		else if (root instanceof DeclarationStatement)
		{
			// need to skip because comments are DeclarationStatement
		}
		else
		{
			FlatIterator iter = new FlatIterator(root);

			while (iter.hasNext())
			{
				Object obj = iter.next();

				addWriteAccesses(loop, loopArrayAccessMap, (Traversable)obj);
			}
		}
	}

	private void addReadAccesses(Loop loop,
			HashMap<Symbol, ArrayList<DDArrayAccessInfo>>loopArrayAccessMap,
			Traversable root)
	{
		BreadthFirstIterator iter = new BreadthFirstIterator(root);
		iter.pruneOn(AccessExpression.class);
		iter.pruneOn(AssignmentExpression.class);
		iter.pruneOn(Loop.class);

		HashSet<Class> of_interest = new HashSet<Class>();
		of_interest.add(AccessExpression.class);
		of_interest.add(ArrayAccess.class);
		of_interest.add(AssignmentExpression.class);
		of_interest.add(Loop.class);

		for (;;)
		{
			Object o = null;

			try {
				o = iter.next(of_interest);
			} catch (NoSuchElementException e) {
				break;
			}

			if (o instanceof AccessExpression)
			{
				AccessExpression expr = (AccessExpression)o;

				/*
				 *  The left-hand side of an access expression
				 *  is read in the case of p->field.  For accesses
				 *  like p.field, we still consider it to be a use
				 *  of p because it could be a use in C++ or Java
				 *  (because p could be a reference) and it doesn't
				 *  matter for analysis of C (because it will never
				 *  be written.
				 */
				addReadAccesses(loop, loopArrayAccessMap, expr.getLHS());
			}
			else if (o instanceof AssignmentExpression)
			{
				AssignmentExpression expr = (AssignmentExpression)o;

				/* Recurse on the right-hand side because it is being read. */
				addReadAccesses(loop, loopArrayAccessMap, expr.getRHS());

				/*
				 *  The left-hand side also may have uses, but unless the
				 *  assignment is an update like +=, -=, etc. the top-most
				 *  left-hand side expression is a definition and not a use.
				 */
				if (expr.getOperator() != AssignmentOperator.NORMAL)
				{
					addReadAccesses(loop, loopArrayAccessMap, expr.getLHS());
				}
			}
			else if (o instanceof Loop)
			{
			}
			else
			{
				Statement stmt = ((Expression)o).getStatement();
				DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
						(ArrayAccess)o, DDArrayAccessInfo.read_type, loop, stmt);
				addArrayAccess(arrayInfo, loopArrayAccessMap);
			}
		}
	}

	private void addArrayAccess(DDArrayAccessInfo info,
			HashMap<Symbol, ArrayList<DDArrayAccessInfo>>loopArrayAccessMap)
	{

		Symbol arrayName = Tools.getSymbolOf((info.getArrayAccess()));
		if (loopArrayAccessMap.containsKey(arrayName))
		{
			(loopArrayAccessMap.get(arrayName)).add(info);
		}
		else
		{
			ArrayList<DDArrayAccessInfo> infoList = new ArrayList<DDArrayAccessInfo>();
			infoList.add(info);
			loopArrayAccessMap.put(arrayName, infoList);
		}
	}

	/**
	 * runDDTest uses framework information collected by the DDTDriver pass to obtain a pair of 
	 * array accesses and pass them to the DDTest calling interface.
	 */
	private DDGraph runDDTest(
			Loop loop,
			HashMap<Symbol, ArrayList<DDArrayAccessInfo>>loopArrayAccessMap,
			HashMap<Loop, LoopInfo> loopInfoMap)
	{
		/* Dependence graph to hold direction vectors for current loop nest */
		DDGraph loopDDGraph = new DDGraph(); 
		
		/* Storage of information related to privatization, reduction and alias analyses */
		Set<Symbol> private_set = new HashSet<Symbol>();
		HashMap<Loop, Symbol> reduction_set = new HashMap<Loop, Symbol>();
		Set<Symbol> alias_set = new HashSet<Symbol>();
		for (Loop l: loopInfoMap.keySet())
		{
			Set<Symbol> s = (Set)(Tools.getAnnotation((Statement)l, "cetus", "private"));
			if (s != null)
				private_set.addAll(s);
			
			Map<String, Set<Expression>> m = 
				(Map<String, Set<Expression>>)(Tools.getAnnotation((Statement)l, "cetus", "reduction"));
			if (m != null)
			{
				for (String op : m.keySet())
				{
					Set<Expression> ts = (Set<Expression>)m.get(op);
					for (Expression e: ts)
					{
						reduction_set.put(l, (Tools.getSymbolOf(e)));
					}
				}
			}
		}
		/* Run Alias Analysis as currently it has been implemented as whole program
		 * analysis
		 */
		AliasAnalysis alias_analysis = new AliasAnalysis(program);
		AnalysisPass.run(alias_analysis);
		
		/* SCALAR DEPENDENCE CHECK
		 * -----------------------
		 * Currently, skip array data dependence testing for loops that contain scalar
		 * variables in their def set that are not marked private or reduction. These scalars 
		 * may cause dependences that we don't currently test for.
		 * Consider these loops serial by returning null for the loop dependence graph
		 */
		Set<Symbol> def_symbols = Tools.getDefSymbol((Statement)loop.getBody());
		for (Symbol s : def_symbols)
		{
			/* If the variable is a scalar that is written to, and not marked private 
			 * or reduction
			 */
			if ((Tools.isScalar(s)) && 
				(!(private_set.contains(s))
				&& !(reduction_set.containsValue(s)))
				) {
				System.out.println(s);
//				return null;
			}
		}
		
		/* Get all the array names from the loopArrayAccessMap
		 * For each name, a list of accesses is obtained and a pair of accesses such that at least
		 * one is of 'DDArrayAccessInfo.write_type' is passed to DDTest for dependence analysis
		 */
		Set<Symbol> arrayNames = loopArrayAccessMap.keySet();
		for (Symbol name: arrayNames)
		{
			System.out.println(name);
			ArrayList<DDArrayAccessInfo> arrayList = loopArrayAccessMap.get(name);
			ArrayList<DDArrayAccessInfo> arrayList2 = loopArrayAccessMap.get(name);
			
			/* -------------------
			 * PRIVATIZATION FILTER
			 */
			if (!(private_set.isEmpty()) && private_set.contains(name))
			{
				Tools.println("Skipping over private array variable: " + name.getSymbolName(), 2);
				continue;
			}
			/* -------------------
			 * REDUCTION VARIABLE FILTER
			 */
			if (!(reduction_set.isEmpty()) && reduction_set.containsValue(name))
			{
				Tools.println("Skipping over reduction variable: " + name.getSymbolName(), 2);
				continue;
			}
			/* -------------------
			 * ALIAS SET CHECK
			 */
			alias_set = alias_analysis.get_alias_set((Statement)loop, name);
			if ((alias_set != null) && !(alias_set.isEmpty()))
			{
				for (Symbol s: alias_set)
				{
					if (loopArrayAccessMap.containsKey(s))
						arrayList2.addAll(loopArrayAccessMap.get(s));
				}
			}

			Iterator<DDArrayAccessInfo> iter_expr1 = arrayList.iterator();
			while(iter_expr1.hasNext())
			{
				/* Iterate over all the write accesses using iter_expr1 */
				DDArrayAccessInfo expr1_info = iter_expr1.next();
				System.out.println(expr1_info);
				if (expr1_info.getAccessType() == DDArrayAccessInfo.write_type)
				{
					Iterator<DDArrayAccessInfo> iter_expr2 = arrayList2.iterator();
					while (iter_expr2.hasNext())
					{
						/* Iterate over all accesses with the same name or aliased using iter_expr2 */
						DDArrayAccessInfo expr2_info = iter_expr2.next();
						System.out.println(expr2_info);

						Tools.println("Testing the pair: expr1 - " + expr1_info.toString() + 
								" expr2 - " + expr2_info.toString(), 2);

						
						LinkedList<Loop> nest = LoopTools.getCommonNest(expr1_info.getAccessLoop(), 
																	expr2_info.getAccessLoop(), 
																	loopInfoMap);
						/* Create DDTest with the common nest for the two array accesses being tested */
						DDTest ddt = new DDTest(nest,loopInfoMap);
						ArrayList<DependenceVector> DVset = new ArrayList<DependenceVector>();
						/* Pass pair of accesses to dependence test and store resulting
						 * direction vector set in DVset
						 */
						int depExists = ddt.testAccessPair(expr1_info.getArrayAccess(), 
								expr2_info.getArrayAccess(), 
								DVset);
						System.out.println(depExists);
						Tools.println ("Dependence vectors:", 2);
						for (DependenceVector dv: DVset)
						{
							if (dv.valid)
								Tools.println(dv.VectorToString(), 2);
						}

						/* For every direction vector in the set, add a dependence arc in 
						 * the loop data dependence graph if depExists
						 */
						if (depExists == 1)
						{
							Iterator <DependenceVector> iter_DVset = DVset.iterator();
							while (iter_DVset.hasNext())
							{
								DependenceVector DV = iter_DVset.next();
								DDGraph.Arc arc = new DDGraph.Arc(expr1_info, expr2_info, DV);
								loopDDGraph.addArc(arc);
							}
						}
						else if (depExists == 0)
						{
							Tools.println("No dependence between the pair", 2);
						}
						/* DDTest found the pair of accesses to contain non-affine subscripts
						 * Return loopDDGraph = null which would cause the driver to skip this
						 * loop and proceed to the next
						 */
						else if (depExists == 2)
						{
							return null;
						}
					}
				}
			}
		}
		
		loopDDGraph.removeDuplicateArcs();
		Tools.println(loopDDGraph.toString(), 2);
		return loopDDGraph;		
	}
}
