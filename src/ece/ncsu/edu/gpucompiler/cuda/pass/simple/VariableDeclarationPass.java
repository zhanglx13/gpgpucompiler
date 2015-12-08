package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.List;

import cetus.hir.DepthFirstIterator;
import cetus.hir.Procedure;
import cetus.hir.Tools;
import cetus.hir.VariableDeclaration;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class VariableDeclarationPass extends Pass {

	
	
	public VariableDeclarationPass() {
		
	}
	
	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}


	@Override
	public void dopass(GProcedure proc) {
		Procedure procedure = proc.getProcedure();
		DepthFirstIterator dfi = new DepthFirstIterator(procedure);
		List<VariableDeclaration> decs = dfi.getList(VariableDeclaration.class);
		for (VariableDeclaration vd: decs) {
			Tools.addSymbols(procedure, vd);
		}		
	}

}
