package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.NoSuchElementException;

import cetus.hir.BreadthFirstIterator;
import cetus.hir.CompoundStatement;
import cetus.hir.Statement;
import cetus.hir.Traversable;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class CompoundStatementPass extends Pass {



	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}

	@Override
	public void dopass(GProcedure proc) {
		boolean haschange = true;
		while (haschange) {
			haschange = false;
			BreadthFirstIterator iter = new BreadthFirstIterator(proc.getProcedure());

			for (;;) {
				try {
					Statement st = (Statement) iter.next(Statement.class);
					if (st instanceof CompoundStatement) {
						CompoundStatement cs = (CompoundStatement) st;
						Traversable parent = cs.getParent();
						if (parent!=null&&(parent instanceof CompoundStatement)) {
							CompoundStatement pcs = (CompoundStatement)parent;
							for (Traversable child : cs.getChildren()) {
								Statement stmt = (Statement)child;
								pcs.addStatementBefore(cs, (Statement)stmt.clone());
							}
							cs.detach();
							haschange = true;
						}
					}
					else 
					if (st.toString().trim().equals("")) {
						st.detach();
						haschange = true;
					}
				} catch (NoSuchElementException e) {
					break;
				}
			}			
		}
	}

}
