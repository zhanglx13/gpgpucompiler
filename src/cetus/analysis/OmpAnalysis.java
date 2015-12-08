package cetus.analysis;

import java.util.*;
import cetus.hir.*;
import cetus.exec.*;

/**
	* This pass analyzes openmp pragmas and converts them into Cetus Annotations, 
	* the same form as what parallelization passes generate. The original OpenMP 
	* pragmas are removed after the analysis.
	*/
public class OmpAnalysis extends AnalysisPass
{
	private int debug_level;

	/** 
		* omp_map: HashMap with an OpenMP construct as a key and a list of OpenMP HashMaps
		*/
	private	HashMap omp_map;

	public OmpAnalysis(Program program)
	{
		super(program);
		debug_level = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
	}

	public String getPassName()
	{
		return new String("[Omp]");
	}

	private void check_omp_independent(int omp_pragma_id)
	{
	}

	public void start()
	{
		Annotation new_annot = null;
		boolean attach_omp_to_nxt_stmt = false;
		omp_map = null;

		if (debug_level >= 2) System.out.println("[Omp.java] Start");
		/* iterate over everything, with particular attention to annotations */
		DepthFirstIterator iter = new DepthFirstIterator(program);

		while(iter.hasNext())
		{
			Object obj = iter.next();

			if (obj instanceof Annotation)
			{
				// Each directive starts with #pragma omp. The remainder of the directive 
				// follows the conventions of the C and C++ standards for compiler directives. 
				// In particular, white space can be used before and after the #, and sometimes 
				// white space must be used to separate the words in a directive. - [OpenMP3.0]

				String pragma = ((Annotation)obj).getText();

				new_annot = new Annotation("cetus");
				omp_map = (HashMap)(new_annot.getMap());

				pragma = pragma.replace("#pragma", "# pragma");

				// The delimiter for split operation is white space(\s).
				// Parenthesis, comma, and colon are delimiters, too. However, we want to leave 
				// them in the pragma token array. Thus, we append a space before and after the
				// parenthesis and colons so that the split operation can recognize them as
				// independent tokens.

				pragma = pragma.replace("(", " ( ");
				pragma = pragma.replace(")", " ) ");
				pragma = pragma.replace(":", " : ");
				pragma = pragma.replace(",", " , ");

				String [] token_array = pragma.split("\\s+");
				if (	token_array[0].compareTo("#")==0 &&
							token_array[1].compareTo("pragma")==0 &&
							token_array[2].compareTo("omp")==0  )
				{
					// omp_parser.run puts the OpenMP directive parsing results into omp_map
					attach_omp_to_nxt_stmt = OmpParser.parse_omp_pragma(omp_map, token_array);

					if (!attach_omp_to_nxt_stmt)
					{

						/** cur_stmt is the DeclarationStatement that has OpenMP pragma */
						Statement cur_stmt = (Statement)((Declaration)obj).getParent();

						/** find the parent_stmt and insert the new Anntation statement */
						Statement parent_stmt = (Statement)(((DeclarationStatement)cur_stmt).getParent());
						AnnotationStatement annot_stmt = new AnnotationStatement(new_annot);
						/** attach the new openmp annotationStatement as attachedStatement to itself */
						annot_stmt.attachStatement(annot_stmt);
          	((CompoundStatement)parent_stmt).addStatementBefore(cur_stmt, annot_stmt);
					}
				}
			}
			else if (obj instanceof Statement) { 	// the very next statement to attach the OmpDirective 
				if (attach_omp_to_nxt_stmt) {
					Statement cur_stmt = (Statement)obj;
					if (debug_level >= 2) 
					{
						System.out.println("[Omp] inserting omp_map to ...");
						System.out.println("------------------------------");
						((Statement)obj).print(System.out);
						System.out.println("------------------------------");
					}

					if (new_annot != null)
					{
						// Insert a Statement and a corresponding OmpDirective pair

						/** find the parent_stmt and insert the new Anntation statement */
						Statement parent_stmt = (Statement)(cur_stmt.getParent());
						AnnotationStatement annot_stmt = new AnnotationStatement(new_annot);
						/** attach the new cetus Annotation to the cur_stmt */
						annot_stmt.attachStatement(cur_stmt);
 	        	((CompoundStatement)parent_stmt).addStatementBefore(cur_stmt, annot_stmt);
					} else
					{
						System.out.println("Error");
						System.exit(0);
					}

					// clear the omp directive flag
					attach_omp_to_nxt_stmt = false;
				}
			}
		}

		/**
			*	Debugging
			*/
		if ( debug_level >= 2 ) display();

		/**
			*	Shared Data Analysis
			*/
		shared_analysis(program);

		if (debug_level >= 2) System.out.println("[Omp] analysis complete");
	}

	public void shared_analysis(Program program)
	{
		if (debug_level >= 2) System.out.println("shared_anlaysis strt");

		FlatIterator iter = new FlatIterator(program);
		/** we assume that the annotations cannot be nested */
		Set annot_set = iter.getSet(AnnotationStatement.class);

		for (AnnotationStatement annot_stmt : (Set<AnnotationStatement>)(annot_set))
		{
			Annotation annot = annot_stmt.getAnnotation();

			HashMap map = (HashMap)(annot.getMap());
			if ( map.keySet().contains("parallel") )
			{
				HashSet<String> OmpSharedSet = null, OmpPrivSet = null;
				HashSet<String> SharedSet;

				// shared variables explicitly defined by the OpenMP directive
				if ( map.keySet().contains("shared") )
				{
					OmpSharedSet = (HashSet)map.get("shared");
				}

				// private variables explicitly defined by the OpenMP directive
				if ( map.keySet().contains("private") )
				{
					OmpPrivSet = (HashSet)map.get("private");
				}

				boolean default_is_shared = true;
				if ( map.keySet().contains("default") )
				{
					String default_value = (String)(map.get("default"));
					if ( default_value.equals("none") ) default_is_shared = false;
				}

				Statement target_stmt = annot_stmt.getStatement();
				SharedSet = find_shared_set(target_stmt, OmpSharedSet, OmpPrivSet, default_is_shared);	
			}
		}

		if (debug_level >= 2) System.out.println("shared_anlaysis done");
	}

	public HashSet<String> find_shared_set(Statement stmt, HashSet<String> OmpSharedSet, HashSet<String> OmpPrivSet, boolean default_is_shared)
	{
		Set<String> LocalSet = null;
		HashSet<String> UseSet, DefSet;

		if (debug_level >= 2) 
			System.out.println("Performing find_shared_set to " + stmt.getClass().getName());

		/**
			* For ArrayAccess case, Tools.getDefSet and Tools.getUseSet returns only 
			* the ArrayAccess, for example, "A[i]=...", Def set has only A[i], not A
			*/
		DefSet = convert2SetOfStrings( Tools.getDefSet(stmt) );
		UseSet = convert2SetOfStrings( Tools.getUseSet(stmt) );

		// For loops, LocalSet only contains the loop variables, 
		// loop body also needs to be checked for local variables 
		if (stmt instanceof ForLoop )
		{
			HashSet<String> LoopBodyLocal;
			Statement loopbody = ((ForLoop)stmt).getBody();
			LocalSet = convert2SetOfStrings( Tools.getSymbols((SymbolTable)loopbody));
		}
		else if (stmt instanceof CompoundStatement)
		{
			// local variables declared locally within this parallel region
			LocalSet = convert2SetOfStrings( Tools.getSymbols((SymbolTable)stmt));
		}
		else 
		{
			// do nothing for a function call
		}

		// ipaSharedSet is a set of shared variables in the functions called within
		// the current scope
		HashSet<String> ipaSharedSet = new HashSet<String>();
		BreadthFirstIterator iter = new BreadthFirstIterator((Traversable) stmt);
		for (;;)
		{
			FunctionCall call = null;

			try {
				call = (FunctionCall)iter.next(FunctionCall.class);
			} catch (NoSuchElementException e) {
				break;
			}

			/* called_procedure is null for system calls */
			Procedure called_procedure = call.getProcedure();
			HashSet<String> ProcedureSharedSet = null;
			if (called_procedure != null)
			{	/* recursive call to find_shared_set routine */
				if (debug_level >= 2) 
					System.out.println("Performing IPA into the procedure: " + called_procedure.getName());
				ProcedureSharedSet = find_shared_set(called_procedure.getBody(), OmpSharedSet, OmpPrivSet, default_is_shared);
				if (ProcedureSharedSet != null) {
					displaySet("ProcedureSharedSet in " + called_procedure.getName(),  ProcedureSharedSet);
					ipaSharedSet.addAll(ProcedureSharedSet);
				}
			}
		}

		HashSet<String> SharedSet = new HashSet<String>();

		// (default==shared) SharedSet = UseDef + ipaShared + OmpShared - OmpPriv - Local
		// (default==none)   SharedSet = ipaShared + OmpShared - OmpPriv - Local
		if (default_is_shared)
		{
			if (UseSet != null) SharedSet.addAll(UseSet);
			if (DefSet != null) SharedSet.addAll(DefSet);
		}
		if (ipaSharedSet != null) SharedSet.addAll(ipaSharedSet);
		if (OmpSharedSet != null) SharedSet.addAll(OmpSharedSet);
		if (OmpPrivSet != null)		SharedSet.removeAll(OmpPrivSet);
		if (LocalSet != null)			SharedSet.removeAll(LocalSet);

		displaySet("DefSet", DefSet);
		displaySet("UseSet", UseSet);
		displaySet("ipaSharedSet", ipaSharedSet);
		displaySet("OmpSharedSet", OmpSharedSet);
		displaySet("OmpPrivSet", OmpPrivSet);
		displaySet("LocalSet", LocalSet);
		displaySet("Final SharedSet", SharedSet);

		return SharedSet;
	}

	/**
		*	Implicit barrier
		* 	- at the end of the parallel construct
		* 	- at the end of the worksharing construct (check an existence of nowait clause)
		* 	- at the end of the sections construct (check an existence of nowait clause)
		* 	- at the end of the single construct (check an existence of nowait clause)
		*
		*/
	public void mark_interval()
	{
		if ( debug_level >= 2 ) System.out.println("[mark_interval] strt");

		FlatIterator iter = new FlatIterator(program);
		/** we assume that the annotations cannot be nested */
		Set annot_set = iter.getSet(AnnotationStatement.class);

		for (AnnotationStatement annot_stmt : (Set<AnnotationStatement>)(annot_set))
		{
			Annotation annot = annot_stmt.getAnnotation();
		
			if ( Tools.containsAnnotation(annot, "cetus", "parallel") )
			{
			}
		}

		if ( debug_level >= 2 ) System.out.println("[mark_interval] done");
	}

	/**
		*	This method is for debugging purpose; it shows the statement that
		*	has an OpenMP pragma.
		*/
	public void display()
	{
		FlatIterator iter = new FlatIterator(program);
		/** we assume that the annotations cannot be nested */
		Set annot_set = iter.getSet(AnnotationStatement.class);

		for (AnnotationStatement annot_stmt : (Set<AnnotationStatement>)(annot_set))
		{
			Annotation annot = annot_stmt.getAnnotation();
			HashMap map = (HashMap)(annot.getMap());

			((Statement)annot_stmt).print(System.out);
			System.out.println("has the following OpenMP pragmas");
			for ( String ikey : (Set<String>)(map.keySet()) )
			{	/* (key, set) pairs */
				if (ikey.compareTo("private")==0 ||
						ikey.compareTo("firstprivate")==0 ||
						ikey.compareTo("lastprivate")==0 ||
						ikey.compareTo("shared")==0 ||
						ikey.compareTo("copyprivate")==0 ||
						ikey.compareTo("copyin")==0 ||
						ikey.compareTo("threadprivate")==0 ||
						ikey.compareTo("flush")==0 )
				{
					System.out.print("[" + ikey + "] (");
					displaySet( (Set<String>)(map.get(ikey)) );
					System.out.println(")");
				}
				else if (ikey.compareTo("critical")==0 ||
								 ikey.compareTo("if")==0 ||
								 ikey.compareTo("num_threads")==0 ||
								 ikey.compareTo("schedule")==0 ||
								 ikey.compareTo("collapse")==0 ||
								 ikey.compareTo("default")==0 )
				{	/* (key, string) pairs */
					System.out.print("[" + ikey + "] (");
					System.out.print( map.get(ikey) );
					System.out.println(")");
				}
				else if (ikey.compareTo("reduction")==0)
				{	/* ("reduction", reduction_map) pair */ 
					HashMap reduction_map = (HashMap)(map.get(ikey));
					System.out.print("[" + ikey + "] ");
					for (String op : (Set<String>)(reduction_map.keySet()) )
					{
						System.out.print("(" + op + ": ");
						displaySet( (Set<String>)(reduction_map.get(op)) );
						System.out.print(")");
					}
					System.out.println();
				}
				else {	/* (key, "true") pairs */
					System.out.println("[" + ikey + "]");
				}
			}
		}
	}

	static public void displayList(LinkedList<String> list)
	{
		int cnt = 0;
		for (String ilist : list)
		{
			if ( (cnt++)!=0 ) System.out.print(", ");
			System.out.print(ilist);
		}
	}

	public void displaySet(Set<String> iset)
	{
		int cnt = 0;
		if (iset == null) return;
		for ( String is : iset )
		{
			if ( (cnt++)!=0 ) System.out.print(", ");
			System.out.print(is);
		}
	}

	public void displaySet(String name, Set<String> iset)
	{
		int cnt = 0;
		if (iset == null) return;
		System.out.print(name + ":");
		for ( String is : iset )
		{
			if ( (cnt++)!=0 ) System.out.print(", ");
			System.out.print(is);
		}
		System.out.println("\n");
	}

	public HashSet convert2SetOfStrings( Set iset )
	{
		HashSet oset = new HashSet();
		if (iset == null) return null;
		for ( Object obj : iset ) 
		{ 
			if (obj instanceof Symbol)					oset.add(((Symbol)obj).getSymbolName());
			else if (obj instanceof Expression) oset.add(((Expression)obj).toString());
			else 																System.out.println("Error");
		}
		return oset;
	}

}

/*
	// Insert an annotation
	for (HashMap note : dir) 
	{
		Annotation.printAsPragma(note, System.out);

		Traversable old_annot = ((Declaration)obj).getParent();
		CompoundStatement old_annotation = (CompoundStatement)(old_annot.getParent());
		Statement old_annot_stmt = (Statement)old_annot;

		DeclarationStatement new_annot = new DeclarationStatement(note);
		Statement new_annot_stmt = (Statement)new_annot;

		((CompoundStatement)old_annotation).addStatementAfter(old_annot_stmt, new_annot_stmt);
	}
*/

	
