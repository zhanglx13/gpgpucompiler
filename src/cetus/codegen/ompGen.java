package cetus.codegen;

import cetus.hir.*;
import java.util.*;

/**
 * This pass looks for Annotations that provide
 * enough information to add OpenMP pragmas and
 * then inserts those pragmas.
 */
public class ompGen extends CodeGenPass
{
  public ompGen(Program program)
  {
    super(program);
  }

  public String getPassName()
  {
    return new String("[ompGen]");
  }

	public void start()
	{
		DepthFirstIterator iter = new DepthFirstIterator(program);
		ArrayList<ForLoop> loops = iter.getList(ForLoop.class);

		for (ForLoop loop : loops)
		{
			genOmpParallelLoops(loop);
		}
	}

	private void genOmpParallelLoops(ForLoop loop)
	{
		String omp = new String();

		// currently, we check only omp parallel for construct
		if ( !Tools.containsAnnotation(loop, "cetus", "parallel") )
			return;

		Annotation omp_annot = new Annotation("openmp");
		Map omp_map = omp_annot.getMap();

		Set<AnnotationStatement> annot_stmts = Tools.getAnnotStatementSet(loop, "cetus"); 

		for (AnnotationStatement annot_stmt : annot_stmts)
		{
			Annotation annot = annot_stmt.getAnnotation();
			omp_map.putAll(annot.getMap());
		}
		omp_map.put("for", "true");

		omp_annot.setPrintMethod(Annotation.print_as_pragma_method);

		/** insert the new AnntationStatement */
		Statement parent_stmt = (Statement)(loop.getParent());

		AnnotationStatement annot_stmt = new AnnotationStatement(omp_annot);
		((CompoundStatement)parent_stmt).addStatementBefore(loop, annot_stmt);

		/** attach the new openmp AnnotationStatement to the ForLoop */
		annot_stmt.attachStatement(loop);

	}
}
