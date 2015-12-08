package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.Annotation;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Tools;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

public class StatementUtil {
	

	
	public static DeclarationStatement loadDefineShareMemoryStatement(Identifier id, int sizex, int sizey, Specifier spe) {
		List<IntegerLiteral> indexs = new ArrayList<IntegerLiteral>();
		if (sizey>0) {
			indexs.add(new IntegerLiteral(sizey));
		}
		indexs.add(new IntegerLiteral(sizex));
		ArraySpecifier array = new ArraySpecifier(indexs);
		VariableDeclarator declarator = new VariableDeclarator(id, array);
		ArrayList<Specifier> specs = new ArrayList<Specifier>();
		specs.add(Specifier.SHARED);
		specs.add(spe);
		VariableDeclaration dec = new VariableDeclaration(specs, declarator);
		DeclarationStatement stmt = new DeclarationStatement(dec);
		return stmt;
	}
	

	public static ForLoop loadSimpleLoop(Statement stmt, Identifier it, int start, int inc, int end) {
		Statement init = new ExpressionStatement(new AssignmentExpression((Identifier)it.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(start))); 
		Expression condition = new BinaryExpression((Identifier)it.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(end));
        Expression step = new AssignmentExpression((Identifier)it.clone(), AssignmentOperator.NORMAL, 
        		new BinaryExpression((Identifier)it.clone(), BinaryOperator.ADD, new IntegerLiteral(inc)));
        Statement body = (Statement)stmt.clone();
		ForLoop forloop = new ForLoop(init, condition, step, body);
		return forloop;
		
	}
	
	public static DeclarationStatement loadIntDeclaration(Identifier id) {
		VariableDeclarator declarator = new VariableDeclarator((Identifier)id.clone());
		ArrayList<Specifier> specs = new ArrayList<Specifier>();
		specs.add(Specifier.INT);
		VariableDeclaration dec = new VariableDeclaration(specs, declarator);
		DeclarationStatement stmt = new DeclarationStatement(dec);
		return stmt;
		
	}

	public static DeclarationStatement loadInitStatment(Identifier id, Specifier spe) {
		VariableDeclarator declarator = new VariableDeclarator((Identifier)id.clone());
		ArrayList<Specifier> specs = new ArrayList<Specifier>();
		specs.add(spe);
		VariableDeclaration dec = new VariableDeclaration(specs, declarator);
		DeclarationStatement stmt = new DeclarationStatement(dec);
		return stmt;
		
	}
	
	public static DeclarationStatement loadInitStatment(Identifier id, List<Specifier> specs) {
		VariableDeclarator declarator = new VariableDeclarator((Identifier)id.clone());
		VariableDeclaration dec = new VariableDeclaration(specs, declarator);
		DeclarationStatement stmt = new DeclarationStatement(dec);
		return stmt;
		
	}

	
	public static DeclarationStatement updateName(VariableDeclaration sd, String newname) {
		DeclarationStatement ds = (DeclarationStatement)DeclarationUtil.getStatement(sd).clone(); 
		Declaration decl = ds.getDeclaration();
		if (decl instanceof VariableDeclaration) {
			VariableDeclaration vd = (VariableDeclaration)decl;
			VariableDeclarator declarator = (VariableDeclarator)vd.getDeclarator(0);
			VariableDeclarator newv = new VariableDeclarator(new Identifier(newname), declarator.getTrailingSpecifiers());
			newv.setInitializer(declarator.getInitializer());
			vd.setChild(0, newv);
		}
		return ds;
	}
	

	
	public static List<String> parseVariableList(Expression expression) {
		List<String> list = new ArrayList<String>();
		if (expression instanceof BinaryExpression) {
			BinaryExpression be = (BinaryExpression)expression;
			list.addAll(parseVariableList(be.getLHS()));
			list.addAll(parseVariableList(be.getRHS()));
		}
		else
		if (expression instanceof ArrayAccess){
			list.add(((ArrayAccess) expression).getArrayName().toString());
		}
		else
		if (expression instanceof IDExpression){
			list.add(((IDExpression) expression).toString());
		}
		
		return list;
	}
	

	public static Statement createSynchthreadsStatement() {
		return new ExpressionStatement(new FunctionCall(new Identifier(
				"__syncthreads")));
	}
	
	
	public static Annotation createPragma(String str) {
		Annotation ann = new Annotation();
		ann.setText(str);
		ann.setPrintMethod(Annotation.print_as_pragma_method);
		return ann;
	}

	public static void addSibling(Statement current, Statement newstmt,
			boolean isbefore) {
		CompoundStatement parent = (CompoundStatement) current.getParent();
		if (isbefore)
			parent.addStatementBefore(current, newstmt);
		else
			parent.addStatementAfter(current, newstmt);
	
		if (newstmt instanceof DeclarationStatement) {
			DeclarationStatement ds = (DeclarationStatement) newstmt;
			Tools.addSymbols(current.getProcedure(), ds.getDeclaration());
		}
	
	}

	public static Statement getFirstDeclarationStatement(Statement cs) {
			BreadthFirstIterator bfi = new BreadthFirstIterator(cs);
			List<DeclarationStatement> sts = bfi
					.getList(DeclarationStatement.class);
			for (DeclarationStatement st : sts) {
				if (st.getDeclaration() instanceof Annotation)
					continue;
				return st;
			}
			bfi = new BreadthFirstIterator(cs);
			List<ExpressionStatement> ss = bfi.getList(ExpressionStatement.class);
			for (ExpressionStatement s : ss) {
				return s;
			}
	//		System.out.println(cs);
			return sts.get(0);
		}

	public static void addToFirst(CompoundStatement cs, Statement stmtToAdd) {
		if (cs.getChildren().size()==0) {
			cs.addStatement(stmtToAdd);
			return;
		}
		Statement firststmtforproc = (Statement) (new FlatIterator(cs).next());
		cs.addStatementBefore(firststmtforproc, stmtToAdd);		
	}	
		
	
}
