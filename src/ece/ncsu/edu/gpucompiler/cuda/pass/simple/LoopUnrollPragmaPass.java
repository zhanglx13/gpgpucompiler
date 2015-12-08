package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.List;

import cetus.hir.Annotation;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class LoopUnrollPragmaPass extends Pass {



	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}


	@Override
	public void dopass(GProcedure proc) {
		DepthFirstIterator dfi = new DepthFirstIterator(proc.getProcedure());
		List<DeclarationStatement> anns = dfi.getList(DeclarationStatement.class);
		for (DeclarationStatement an: anns) {
			if (an.toString().indexOf("#pragma unroll")!=-1)
				an.detach();
		}
//		System.out.println(proc.getProcedure());
		dfi = new DepthFirstIterator(proc.getProcedure());
		List<GLoop> loops = dfi.getList(GLoop.class);
		for (GLoop forloop: loops) {
			Expression ex = forloop.getEnd();
			if (ex instanceof IntegerLiteral) {
				DeclarationStatement as = createUnrollStatement();
				StatementUtil.addSibling(forloop, as, true);
			}
		}
//		System.out.println(proc.getProcedure());
	}
	
	
	DeclarationStatement createUnrollStatement() {
		Annotation ann = new Annotation();
		ann.setText("unroll");
		ann.setPrintMethod(Annotation.print_as_pragma_method);
		DeclarationStatement as = new DeclarationStatement(ann);
		return as;
		
	}

}
