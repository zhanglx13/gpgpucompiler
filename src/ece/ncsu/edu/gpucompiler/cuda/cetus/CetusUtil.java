package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.BinaryExpression;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.Identifier;
import cetus.hir.Procedure;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;

public class CetusUtil {

	public final static String SYMBOL_PREFIX_SHARED = "shared";
	public final static String SYMBOL_PREFIX_ITERATOR = "it";
	public final static String SYMBOL_PREFIX_TMP = "tmp";

	
	public static VariableDeclaration getDeclarationStatement(Procedure proc, String name) {
		DepthFirstIterator dfi = new DepthFirstIterator(proc);
		List<VariableDeclaration> vds = dfi.getList(VariableDeclaration.class);
		for (VariableDeclaration vd: vds) {
			if (vd.getDeclarator(0).getSymbol().toString().equals(name)) return vd;
		}
		return null;
		
	}
	
	public static void replace(Traversable oldst, Traversable newst) {
		Traversable this_parent = oldst.getParent();
		int this_index = -1;

		if (this_parent != null) {
			this_index = Tools.indexByReference(this_parent.getChildren(),
					oldst);
			if (this_index == -1)
				throw new IllegalStateException();
		}

		/* detach both so setChild won't complain */
		newst.setParent(null);
		oldst.setParent(null);

		if (this_parent != null)
			this_parent.setChild(this_index, newst);

	}
	
	public static boolean isContain(Traversable tr, Identifier id) {
		if (tr instanceof Identifier) {
			if (tr.toString().equals(id.toString())) return true;
		}
		
		for (int i = 0; i < tr.getChildren().size(); i++) {
			Traversable child = tr.getChildren().get(i);
				if (isContain(child, id)) return true;
		}
		return false;		
	}	

	public static Expression replaceChild(Expression expression,
			List<Identifier> oldex, List<Expression> newex) {
		if (expression instanceof AssignmentExpression) {
			AssignmentExpression be = (AssignmentExpression) expression;
			Expression nl = replaceChild(be.getLHS(), oldex, newex);
			Expression nr = replaceChild(be.getRHS(), oldex, newex);
			return new AssignmentExpression(nl, be.getOperator(), nr);

		} else if (expression instanceof BinaryExpression) {
			BinaryExpression be = (BinaryExpression) expression;
			Expression nl = replaceChild(be.getLHS(), oldex, newex);
			Expression nr = replaceChild(be.getRHS(), oldex, newex);
			return new BinaryExpression(nl, be.getOperator(), nr);

		} else if (expression instanceof ArrayAccess) {
			ArrayAccess aa = (ArrayAccess) expression;
//			System.out.println("find: " + aa);
			Expression name = aa.getArrayName();

			Expression nname = replaceChild(name, oldex, newex);
			List<Expression> indexs = new ArrayList();

			for (int i = 0; i < aa.getNumIndices(); i++) {
				Expression index = aa.getIndex(i);
				Expression nindex = replaceChild(index, oldex, newex);
				indexs.add(nindex);
			}
			return new ArrayAccess(nname, indexs);

		} else if (expression instanceof Identifier) {
			Identifier id = (Identifier) expression;
			for (int i = 0; i < oldex.size(); i++) {
				Identifier oe = oldex.get(i);
				Expression ne = newex.get(i);
				if (id.toString().equals(oe.toString())) {
					return (Expression) ne.clone();
				}
			}
		} else {
		}
		return (Expression) expression.clone();

	}

	public static Traversable replaceChild(Traversable tr, Expression oldex,
			Expression newex) {
		if (((Object) tr).getClass().equals(oldex.getClass())) {
			Expression extr = (Expression) tr;
			if (extr.equals(oldex))
				return (Expression) newex.clone();
		}
		for (int i = 0; i < tr.getChildren().size(); i++) {
			Traversable child = tr.getChildren().get(i);
			try {
				tr.setChild(i, replaceChild(child, oldex, newex));
			} catch (Exception ex) {
				// ex.printStackTrace();
			}
		}
		return tr;
	}

//	public static Traversable replaceChild(Traversable tr, Statement oldex,
//			Statement newex) {
//		if (((Object) tr).getClass().equals(oldex.getClass())) {
//			Statement extr = (Statement) tr;
//			if (extr.equals(oldex))
//				return (Statement) newex.clone();
//		}
//		for (int i = 0; i < tr.getChildren().size(); i++) {
//			Traversable child = tr.getChildren().get(i);
//			try {
//				tr.setChild(i, replaceChild(child, oldex, newex));
//			} catch (Exception ex) {
//
//			}
//		}
//		return tr;
//	}

//	public static CompoundStatement unroll(GLoop loop, GProcedure func) {
//		try {
//			GLoop loopstmt = new GLoop(loop, func);
//			CompoundStatement cs = new CompoundStatement();
//			if (loopstmt.getEnd() instanceof IntegerLiteral) {
//				IntegerLiteral end = (IntegerLiteral) loopstmt.getEnd();
//				for (long i = loopstmt.getStart(); i < end.getValue(); i += loopstmt
//						.getIncrement().getValue()) {
//					Statement stmt = (Statement) replaceChild(
//							(Statement) loopstmt.getBody().clone(), loopstmt
//									.getIterator(), new IntegerLiteral(i));
//					cs.addStatement(stmt);
//				}
//			} else {
//				return null;
//			}
//			return cs;
//		} catch (UnsupportedCodeException ex) {
//			ex.printStackTrace();
//		}
//		return null;
//	}

	public static int getSmallMultiple(int value, int multiple) {
		while (value % multiple != 0)
			value--;
		return value;
	}


}
