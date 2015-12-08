package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Procedure;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Tools;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.VariableTools;

public class PrefetchingPass extends Pass {

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}

	@Override
	public void dopass(GProcedure proc) {
		// TODO Auto-generated method stub
		DepthFirstIterator dfi = new DepthFirstIterator(proc.getProcedure());
		List<MemoryExpression> list = dfi.getList(MemoryExpression.class);
		
		List<MemoryExpression> pflist = new ArrayList();
		GLoop loop = null;
		for (MemoryExpression me: list) {
			if (loop==null) loop = me.getLoop();
			if (loop==me.getLoop()) {
				pflist.add(me);
			}
		}
		
		filter(proc, pflist);
	}	
	


	public void filter(GProcedure proc, List<MemoryExpression> mss) {
		if (mss==null||mss.size()==0) return;
		MemoryExpression ms = mss.get(0);
		GLoop loop = ms.getLoop();
		Statement synStmt = null;
		Statement cs = ms.getStatement();
		while (synStmt==null)
		{
			if (cs==null) break;
			if (cs.getParent() instanceof Procedure) break;
			cs = (Statement)cs.getParent();
			
			FlatIterator iter = new FlatIterator(cs);
	
		    for (;;)
		    {
		      try {
		    	  Statement stmt = (Statement)iter.next(Statement.class);
		    	  if (stmt.toString().startsWith("__syncthreads")) {
		    		  synStmt = stmt;
		    	  }
		      } catch (NoSuchElementException e) {
		        break;
		      } 
		    }		
//		    System.out.println(synStmt);
		}
	    if (synStmt==null||!(cs instanceof CompoundStatement)) return;
	    System.out.println(cs);
	    
		BinaryExpression cond = new BinaryExpression(loop.getIterator(), BinaryOperator.COMPARE_LT, 
				new BinaryExpression(loop.getEnd(), BinaryOperator.SUBTRACT, loop.getIncrement()));
		Expression newit = new BinaryExpression(ms.getLoop().getIterator(), BinaryOperator.ADD, ms.getLoop().getIncrement());
		CompoundStatement ifcs = new CompoundStatement();
		IfStatement ifs =  new IfStatement(cond, ifcs);
		Identifier[] tempid = new Identifier[mss.size()];
		for (int i=0; i<mss.size(); i++) {
			tempid[i] = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_TMP, proc.getProcedure());
			ms = mss.get(i);
			BinaryExpression be = ms;
			Expression rhs = be.getRHS();
			be.setRHS((Identifier)tempid[i].clone());
			AssignmentExpression newbe = new AssignmentExpression((Identifier)tempid[i].clone(), AssignmentOperator.NORMAL,
					(Expression)CetusUtil.replaceChild((Expression)rhs.clone(), ms.getLoop().getIterator(), newit));
			ifcs.addStatement(new ExpressionStatement(newbe));
			DeclarationStatement dec = StatementUtil.loadInitStatment((Identifier)tempid[i].clone(), Specifier.FLOAT);
			StatementUtil.addSibling(loop, dec, true);
			Statement stmt = new ExpressionStatement(
					new AssignmentExpression((Identifier)tempid[i].clone(), AssignmentOperator.NORMAL, 
							(Expression)CetusUtil.replaceChild(rhs, ms.getLoop().getIterator(), new IntegerLiteral(ms.getLoop().getStart()))));
			if (ms.getStatement().getParent().getParent() instanceof IfStatement) {
				IfStatement pif = (IfStatement)ms.getStatement().getParent().getParent();
				stmt = new IfStatement((Expression)pif.getControlExpression().clone(), stmt);
			}
			StatementUtil.addSibling(loop, stmt, true);
			Tools.addSymbols(proc.getProcedure(), dec.getDeclaration());
		}
		
		if (ms.getStatement().getParent().getParent() instanceof IfStatement) {
			IfStatement pif = (IfStatement)ms.getStatement().getParent().getParent();
			ifs = new IfStatement((Expression)pif.getControlExpression().clone(), ifs);
		}
		((CompoundStatement)cs).addStatementBefore(synStmt, ifs);
		
		
		System.out.println(ms.getLoop().getGProcedure().getProcedure());
	}
}
