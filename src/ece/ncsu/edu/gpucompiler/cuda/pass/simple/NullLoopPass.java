package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.List;

import cetus.hir.AssignmentExpression;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Procedure;
import cetus.hir.Statement;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.DeclarationUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArray;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;
import ece.ncsu.edu.gpucompiler.cuda.cetus.VariableTools;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

/**
 * add or remove meaningless loop
 * @author jack
 *
 */
public class NullLoopPass extends Pass {

	boolean isAdd = true;
	
	public NullLoopPass(boolean isAdd) {
		this.isAdd = isAdd;
	}
	


	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}


	@Override
	public void dopass(GProcedure proc) {
		if (isAdd) {
			addNullLoop(proc);
		}
		else {
			removeNullLoop(proc);
			
		}
	}
	
	void addNullLoop(GProcedure proc) {
		Procedure procedure = proc.getProcedure();
		DepthFirstIterator dfi = new DepthFirstIterator(procedure);
		List<AssignmentExpression> assignmentExpressions = (List<AssignmentExpression>)dfi.getList(AssignmentExpression.class);
		for (AssignmentExpression assignmentExpression: assignmentExpressions) {
			Traversable tr = assignmentExpression.getParent();
			while (!(tr instanceof GLoop)&&tr!=null) {
				tr = tr.getParent();
			}
			if (tr==null) {
		        MemoryArray lhs = MemoryExpression.parse(assignmentExpression.getLHS(), proc);
		        MemoryArray rhs = MemoryExpression.parse(assignmentExpression.getRHS(), proc);
		        boolean goon = false;
		        if (lhs!=null&&lhs.getMemoryType()==MemoryArray.MEMORY_GLOBAL) {
		        	goon = true;
		        }
		        if (rhs!=null&&rhs.getMemoryType()==MemoryArray.MEMORY_GLOBAL) {
		        	goon = true;
		        }
				if (!goon) continue;
				try {
					Identifier it = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_ITERATOR, procedure);
					
					ForLoop forloop = StatementUtil.loadSimpleLoop(assignmentExpression.getStatement(), it, 0, 1, 1);
			        DeclarationStatement ds = StatementUtil.loadIntDeclaration(it);
					GLoop gloop;
						gloop = new GLoop(forloop, proc);
					assignmentExpression.getStatement().swapWith(gloop);
					StatementUtil.addSibling(gloop, ds, true);
					Tools.addSymbols(proc.getProcedure(), ds.getDeclaration());
				} catch (UnsupportedCodeException e) {
					e.printStackTrace();
				}

			}
		}
	}
	


	void removeNullLoop(GProcedure proc) {
		boolean changed = true;
		while (changed) {
			changed = false;
			DepthFirstIterator dfi = new DepthFirstIterator(proc.getProcedure());
			List<GLoop> loops = dfi.getList(GLoop.class);
			for (GLoop forloop: loops) {
				Expression ex = forloop.getEnd();
				if (ex instanceof IntegerLiteral) {
					if (((IntegerLiteral) ex).getValue()<=forloop.getStart()+forloop.getIncrement().getValue()) {
						Identifier it = forloop.getIterator();
						DepthFirstIterator dfi1 = new DepthFirstIterator(proc.getProcedure());
						List<VariableDeclaration> gvds = dfi1.getList(VariableDeclaration.class);
						for (VariableDeclaration gvd:gvds) {
							if (DeclarationUtil.getVariableName(gvd).equals(it.toString())) {
								DeclarationUtil.getStatement(gvd).detach();
							}
						}
//						int pos = forloop.getParent().getChildren().indexOf(forloop);
//						while (pos>0) {
//							Traversable tr = forloop.getParent().getChildren().get(pos-1);
//							if (tr.toString().trim().equals("#pragma unroll")) {
//								System.out.println("previous:"+tr.toString()+tr.getClass());
//								Statement stmt = (Statement)tr;
//								stmt.detach();
//							}
//							else 
//							if (tr.toString().trim().equals("")) {
//								
//							}
//							else {
//								break;
//							}
//							pos--;
//						}
//						
						CetusUtil.replaceChild(forloop.getBody(), forloop.getIterator(), new IntegerLiteral(forloop.getStart()));
						StatementUtil.addSibling(forloop, (Statement)forloop.getBody().clone(), true);
						forloop.detach();
						changed = true;
					}
				}
			}
		}
				
	}
}
