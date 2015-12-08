//###########################################################################//
//             		 gpgpucompiler                                           //
//               Source to Source Compiler for GPGPU (CUDA)                  //
//                                                                           //
//              YI YANG, North Carolina State University                     //
//                            2009-2010                                      //
//###########################################################################//
// Disclaimer:                                                               //
// This code is provided on an "AS IS" basis, without warranty. The author   //
// does not have any liability to you or any other person or entity with     //
// respect to any liability, loss, or damage caused or alleged to have been  //
// caused directly or indirectly by this code.                               //
//###########################################################################//
//###########################################################################//
package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Procedure;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.CompoundStatementPass;
import ece.ncsu.edu.gpucompiler.cuda.util.StringUtil;

public class GProcedureUtil {
	
	
	/**
	 * forward global memory write
	 * @param func
	 * @throws UnsupportedCodeException
	 */
	public static void forwardMemoryWrite(GProcedure func) throws UnsupportedCodeException {

		new CompoundStatementPass().dopass(func);
		func.refresh();
		
		Procedure procedure = func.getProcedure();
		DepthFirstIterator bfi = new DepthFirstIterator(procedure);
		List<AssignmentExpression> ess = bfi.getList(AssignmentExpression.class);

		for (int i = 0; i < ess.size(); i++) {
			AssignmentExpression ae = ess.get(i);
			if (!(ae instanceof MemoryExpression)) continue;
			MemoryExpression es = (MemoryExpression)ae;
//			System.out.print("try to forward "+es);
			if (!es.getLoop().isSimpleLoop()) continue;
			
			if (es.getGlobalMemoryArrayAccess()!=null&&es.getGlobalMemoryArrayAccess()==es.getlMemoryArrayAccess()) {
				Expression ex = es.getRHS();
				Identifier id = null;
				if (ex instanceof Identifier)  {
					id = (Identifier)ex;
				}
				else
				if (ex instanceof ArrayAccess) {
					id = (Identifier)(((ArrayAccess)ex).getArrayName());
				}
				else 
				if (ex instanceof FunctionCall) {
					FunctionCall fc = (FunctionCall)ex;
					if (fc.getNumArguments()==1) {
						if (fc.getArgument(0) instanceof Identifier) {
							id = (Identifier)fc.getArgument(0);
						}
						else 
							continue;
					}
					else 
						continue;
				}
				else continue;
//				System.out.println(id);
				
				Statement asstmt = null;
				for (int j = i - 1; j >= 0; j--) {
					AssignmentExpression nes = ess.get(j);
					Identifier nid = null;
					Expression lhs = nes.getLHS();
					if (lhs instanceof Identifier) 
						nid = (Identifier)lhs;
					else
					if (lhs instanceof ArrayAccess) 
						nid = (Identifier)(((ArrayAccess)lhs).getArrayName());
					else continue;
					
					if (!nid.equals(id)) continue;
						
					Traversable esp = es.getLoop();
					while (esp.getParent()!=null&&(esp.getParent() instanceof CompoundStatement)) {
						esp = esp.getParent();
					}
					Traversable tr = nes;
					while (tr!=null && tr.getParent()!=esp) {
						tr = tr.getParent();
					}
					if (tr!=null&&(tr instanceof Statement)) {
						asstmt = (Statement)tr;
					}
					break;
				}
				if (asstmt != null) {
					StatementUtil.addSibling(asstmt, (Statement) es.getStatement().clone(),
							false);
					es.getStatement().getParent().removeChild(es.getStatement());
				}
			}
		}
	}

	static boolean isloop = false;

	/**
	 * duplicate statements
	 * @param func
	 * @param decsTodo
	 * @param essTodo
	 * @param minNumber
	 * @param maxNumber
	 * @param isX
	 * @param oldid
	 * @param newid
	 */
	public static void duplicateStatement(GProcedure func, List<VariableDeclaration> decsTodo,
				List<ExpressionStatement> essTodo, int minNumber, int maxNumber,
				boolean isX, Identifier oldid, List<Expression> newid) {
		for (ExpressionStatement es : essTodo) {
			if (isloop) 
			{
				try {
					int inc = func.getBlockDimY();
					Identifier it = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_ITERATOR, func.getProcedure());
					Expression nex = GProcedureUtil.updateAsArray(es.getExpression(), decsTodo,
							maxNumber, isX, null, null, inc, it);
					ForLoop forloop = StatementUtil.loadSimpleLoop(new ExpressionStatement(nex), it, 0, 1, maxNumber);
			        DeclarationStatement ds = StatementUtil.loadIntDeclaration(it);
					GLoop gloop;
					gloop = new GLoop(forloop, func);
					es.swapWith(gloop);
					StatementUtil.addSibling(gloop, ds, true);
					Tools.addSymbols(func.getProcedure(), ds.getDeclaration());
				} catch (UnsupportedCodeException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}				
			}
			else {
//				System.out.println("working on"+"----"+minNumber+"-"+maxNumber+"-"+es.toString());
 				CompoundStatement parent = (CompoundStatement) es.getParent();
				List<Expression> exs = GProcedureUtil.update(es.getExpression(), decsTodo,
						maxNumber, isX, oldid, newid, 0);
				boolean find = false;
				for (int i = minNumber; i < maxNumber; i++) {
					ExpressionStatement nes = new ExpressionStatement(exs.get(i));
//					System.out.println(i+nes.toString());
					//if (!nes.toString().equals(es.toString())) 
					{
						parent.addStatementBefore(es, nes);
						find = true;
					}
				}
				if (find)
					parent.removeChild(es);
			}

		}

	}
	

	/**
	 * duplicate statements, and try to use if2array to handle shared memory access
	 * @param func
	 * @param decsTodo
	 * @param essTodo
	 * @param maxNumber
	 * @param isX
	 * @param inc
	 */
	public static void duplicateStatementWithShare(GProcedure func,
				List<VariableDeclaration> decsTodo,
				List<ExpressionStatement> essTodo, int maxNumber, boolean isX,
				int inc) {
		for (ExpressionStatement es : essTodo) {
			CompoundStatement parent = (CompoundStatement) es.getParent();
//			System.out.println("@@@@@@"+parent);
			if (es.getExpression() instanceof AssignmentExpression) {
				AssignmentExpression ae = (AssignmentExpression) es
						.getExpression();
				Expression lhs = ae.getLHS();
				if (lhs instanceof ArrayAccess) {
					ArrayAccess aa = (ArrayAccess) lhs;
					MemoryArray ma = func.getMemoryArray(aa.getArrayName().toString());
					if (ma!=null&&ma.getMemoryType()==MemoryArray.MEMORY_SHARED) {
						if (parent.getParent() instanceof IfStatement) {
							IfStatement ifs = (IfStatement) parent.getParent();
							if (GProcedureUtil.if2array(es, maxNumber, func, ifs))
								continue;
						}

					}
				}
			}

			if (isloop) 
			{
				try {
					Identifier it = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_ITERATOR, func.getProcedure());
					Expression nex = GProcedureUtil.updateAsArray(es.getExpression(), decsTodo,
							maxNumber, isX, null, null, inc, it);
					ForLoop forloop = StatementUtil.loadSimpleLoop(new ExpressionStatement(nex), it, 0, 1, maxNumber);
			        DeclarationStatement ds = StatementUtil.loadIntDeclaration(it);
					GLoop gloop;
					gloop = new GLoop(forloop, func);
					es.swapWith(gloop);
					StatementUtil.addSibling(gloop, ds, true);
				} catch (UnsupportedCodeException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			else {
				List<Expression> exs = GProcedureUtil.update(es.getExpression(), decsTodo,
						maxNumber, isX, inc);
				for (int i = 0; i < maxNumber; i++) {
					parent.addStatementBefore(es, new ExpressionStatement(exs
							.get(i)));
				}
				parent.removeChild(es);
			}

		}

	}

	public static List<Expression> update(Expression expression,
			List<VariableDeclaration> decsTodo, int maxNumber, boolean isX,
			int inc) {
		return GProcedureUtil.update(expression, decsTodo, maxNumber, isX, null, null, inc);
	}

	
	/**
	 * when we expand the thread number in one thread block (for instance, 16->256), 
	 * we also need to duplicate statements. If we can find if statement
	 * we can change 
	 * 	if (tidx<16) sharedM[tidx] = a[idy][idx];
	 * to
	 * 	sharedM[tidx>>4][tidx] = a[16*idy+(tidx>>4)][idx] 
	 * @param es
	 * @param maxNumber
	 * @param func
	 * @param ifs
	 * @return
	 */
	static boolean if2array(ExpressionStatement es, int maxNumber,
				GProcedure func, IfStatement ifs) {

		int tx = func.getBlockDimX();
		BinaryExpression be = (BinaryExpression) ifs.getControlExpression();
		if (!(be.getLHS() instanceof Identifier))
			return false;
		Identifier id = (Identifier) be.getLHS();
		if (!id.toString().equals(ThreadIndex.TIDX))
			return false;
		if (!(be.getRHS() instanceof IntegerLiteral))
			return false;
		IntegerLiteral il = (IntegerLiteral) be.getRHS();
		tx = (int) il.getValue();
		if (tx * maxNumber < func.getBlockDimX()) {
			// multi load statement
			BinaryExpression tidxDivMaxNumber = new BinaryExpression(
					new Identifier(ThreadIndex.TIDX), BinaryOperator.DIVIDE,
					new IntegerLiteral(tx));
			BinaryExpression tidxModuleMaxNumber = new BinaryExpression(
					new Identifier(ThreadIndex.TIDX), BinaryOperator.MODULUS,
					new IntegerLiteral(tx));
//			BinaryExpression nidy = new BinaryExpression(new Identifier(
//					ThreadIndex.IDY), BinaryOperator.MULTIPLY,
//					new IntegerLiteral(maxNumber));
//			nidy = new BinaryExpression(nidy, BinaryOperator.ADD,
//					tidxDivMaxNumber);
			BinaryExpression nidy = new BinaryExpression(new Identifier(
					ThreadIndex.IDY), BinaryOperator.ADD,
					tidxDivMaxNumber);
			List<Identifier> oldex = new ArrayList();
			List<Expression> newex = new ArrayList();
			oldex.add(new Identifier(ThreadIndex.IDY));
			newex.add(nidy);
			oldex.add(new Identifier(ThreadIndex.TIDX));
			newex.add(tidxModuleMaxNumber);
			ArrayList nindexs = new ArrayList();
			Expression nex = CetusUtil.replaceChild(es.getExpression(), oldex, newex);
			ArrayAccess naa = (ArrayAccess) (((AssignmentExpression) nex)
					.getLHS());
			nindexs.add((Expression) naa.getIndex(0).clone());
			nindexs.add((Expression) tidxDivMaxNumber.clone());
			naa.setIndices(nindexs);
			be.setRHS(new IntegerLiteral(tx * maxNumber));
			ifs.setThenStatement(new ExpressionStatement(nex));
			// System.out.println("new expression: "+nex);
			return true;
		} else if (tx * maxNumber == func.getBlockDimX()) {
			// one
			// update tidx->tidx%tx
			// update
			// aa.addIndex(new
			// Identifier(name));

			BinaryExpression tidxDivMaxNumber = new BinaryExpression(
					new Identifier(ThreadIndex.TIDX), BinaryOperator.DIVIDE,
					new IntegerLiteral(tx));
			BinaryExpression tidxModuleMaxNumber = new BinaryExpression(
					new Identifier(ThreadIndex.TIDX), BinaryOperator.MODULUS,
					new IntegerLiteral(tx));
//			BinaryExpression nidy = new BinaryExpression(new Identifier(
//					ThreadIndex.IDY), BinaryOperator.MULTIPLY,
//					new IntegerLiteral(maxNumber));
//			nidy = new BinaryExpression(nidy, BinaryOperator.ADD,
//					tidxDivMaxNumber);
			BinaryExpression nidy = new BinaryExpression(new Identifier(
					ThreadIndex.IDY), BinaryOperator.ADD,
					tidxDivMaxNumber);
			List<Identifier> oldex = new ArrayList();
			List<Expression> newex = new ArrayList();
			oldex.add(new Identifier(ThreadIndex.IDY));
			newex.add(nidy);
			oldex.add(new Identifier(ThreadIndex.TIDX));
			newex.add(tidxModuleMaxNumber);
			ArrayList nindexs = new ArrayList();
			Expression nex = CetusUtil.replaceChild(es.getExpression(), oldex, newex);
			ArrayAccess naa = (ArrayAccess) (((AssignmentExpression) nex)
					.getLHS());
			nindexs.add((Expression) naa.getIndex(0).clone());
			nindexs.add((Expression) tidxDivMaxNumber.clone());
			naa.setIndices(nindexs);
			CompoundStatement pcs = (CompoundStatement) ifs.getParent();
			pcs.addStatementBefore(ifs, new ExpressionStatement(nex));
			pcs.removeChild(ifs);
			return true;

		} else if ((tx * maxNumber) % func.getBlockDimX() == 0) {
			int numberOfStmt = (tx * maxNumber) / func.getBlockDimX();
			// multi load statement
			CompoundStatement pcs = (CompoundStatement) ifs.getParent();
			for (int i = 0; i < numberOfStmt; i++) {
				int offset = func.getBlockDimX() / tx * i;
				BinaryExpression tidxDivMaxNumber = new BinaryExpression(
						new Identifier(ThreadIndex.TIDX),
						BinaryOperator.DIVIDE, new IntegerLiteral(tx));
				BinaryExpression tidxModuleMaxNumber = new BinaryExpression(
						new Identifier(ThreadIndex.TIDX),
						BinaryOperator.MODULUS, new IntegerLiteral(tx));
//				BinaryExpression nidy = new BinaryExpression(new Identifier(
//						ThreadIndex.IDY), BinaryOperator.MULTIPLY,
//						new IntegerLiteral(maxNumber));
//				nidy = new BinaryExpression(nidy, BinaryOperator.ADD,
//						tidxDivMaxNumber);
				BinaryExpression nidy = new BinaryExpression(new Identifier(
						ThreadIndex.IDY), BinaryOperator.ADD,
						tidxDivMaxNumber);
				nidy = new BinaryExpression(nidy, BinaryOperator.ADD,
						new IntegerLiteral(offset));
				List<Identifier> oldex = new ArrayList();
				List<Expression> newex = new ArrayList();
				oldex.add(new Identifier(ThreadIndex.IDY));
				newex.add(nidy);
				oldex.add(new Identifier(ThreadIndex.TIDX));
				newex.add(tidxModuleMaxNumber);
				ArrayList nindexs = new ArrayList();
				Expression nex = CetusUtil.replaceChild(es.getExpression(), oldex, newex);
				ArrayAccess naa = (ArrayAccess) (((AssignmentExpression) nex)
						.getLHS());
				nindexs.add((Expression) naa.getIndex(0).clone());
				nindexs.add((Expression) tidxDivMaxNumber.clone());
				naa.setIndices(nindexs);
//				System.out.println("new expression: " + nex);
				pcs.addStatementBefore(ifs, new ExpressionStatement(nex));
			}
			pcs.removeChild(ifs);
			return true;

		}

		return false;
	}

	/**
	 * duplicate declarations
	 * @param func
	 * @param decsTodo
	 * @param maxNumber
	 */
	public static void duplicateDeclaration(GProcedure func,
				List<VariableDeclaration> decsTodo, int maxNumber) {
		for (VariableDeclaration sd : decsTodo) {
			CompoundStatement parent = (CompoundStatement) DeclarationUtil.getStatement(sd)
					.getParent();
			if (DeclarationUtil.getVariableName(sd).toString().startsWith(CetusUtil.SYMBOL_PREFIX_SHARED)) {
				DeclarationStatement ds =DeclarationUtil.getStatement(sd);
				Declaration decl = ds.getDeclaration();
				if (decl instanceof VariableDeclaration) {
					VariableDeclaration vd = (VariableDeclaration) decl;
					VariableDeclarator declarator = (VariableDeclarator) vd
							.getDeclarator(0);
					if (declarator.getArraySpecifiers().size() == 1) {
						ArraySpecifier spe = (ArraySpecifier) declarator
								.getArraySpecifiers().get(0);
						// IntegerLiteral il =
						// (IntegerLiteral)spe.getDimension(0);
						// il.setValue(il.getValue()+1);
						if (spe instanceof ArraySpecifier) {
							// System.out.println("find share 1d array");
							// map to 2d
							Expression ex = spe.getDimension(0);
							if (ex instanceof IntegerLiteral) {
								IntegerLiteral il = (IntegerLiteral)ex;
								ex = new IntegerLiteral(maxNumber+1);
								spe.setDimension(0, ex);								
								declarator.getArraySpecifiers().add(
									0,
									new ArraySpecifier(new IntegerLiteral(il.getValue())));
							}
							else {
								System.err.println("cannot know the size of shared memory: "+spe);
							}
							continue;
						}
					}
				}
			}
//			System.out.println("handle dec: "+sd);
			if (!isloop) {
				
//				if ("float a;".equals(sd.getStmt().getParent()))
				
//				System.out.println("before:"+sd.getStmt());
				for (int i = 0; i < maxNumber; i++) {
					DeclarationStatement ds0 = StatementUtil.updateName(sd, 
							DeclarationUtil.getVariableName(sd)
							+ "_" + i);
					parent.addStatementBefore(DeclarationUtil.getStatement(sd), ds0);
//					System.out.println("add:"+ds0);
				}
			}
			else {
				DeclarationStatement ds = (DeclarationStatement)DeclarationUtil.getStatement(sd).clone(); 
				Declaration decl = ds.getDeclaration();
				VariableDeclaration vd = (VariableDeclaration)decl;
				List<IntegerLiteral> indexs = new ArrayList<IntegerLiteral>();
				indexs.add(new IntegerLiteral(maxNumber));
				ArraySpecifier array = new ArraySpecifier(indexs);
				VariableDeclarator declarator = new VariableDeclarator(new Identifier(DeclarationUtil.getVariableName(sd)), array);
				List<Specifier> specs = vd.getSpecifiers();
				VariableDeclaration dec = new VariableDeclaration(specs, declarator);
					
				parent.addStatementBefore(DeclarationUtil.getStatement(sd), new DeclarationStatement(dec));
				
			}
			parent.removeChild(DeclarationUtil.getStatement(sd));
		}
	}

	/**
	 * find the decaration and statement need to duplicate, except:
	 * 1. memoryExpressions
	 * 2. for loop
	 * @param func
	 * @param memoryExpressions
	 * @param decsTodo
	 * @param essTodo
	 */
	public static void loadTodos(GProcedure func, List<MemoryExpression> memoryExpressions,
			List<VariableDeclaration> decsTodo,
			List<ExpressionStatement> essTodo) {
	
		List<String> sharedMemories = new ArrayList();
		List<String> variables = new ArrayList();
		List<String> memoryStatments = new ArrayList();
		for (MemoryExpression memoryExpression: memoryExpressions) {
			if (memoryExpression.getSharedMemoryArrayAccess()!=null) {
				sharedMemories.add(memoryExpression.getSharedMemoryArrayAccess().getArrayName().toString());
			}
			else if (memoryExpression.getLHS() instanceof Identifier){
				variables.add(memoryExpression.getLHS().toString());
			}
			memoryStatments.add(memoryExpression.toString());
		}			
		
		DepthFirstIterator dfi = new DepthFirstIterator(func.getProcedure());
		List<MemoryExpression> mss = dfi.getList(MemoryExpression.class);
		for (MemoryExpression memoryExpression: mss) {
			MemoryArrayAccess shared = memoryExpression.getSharedMemoryArrayAccess();
			if (shared==null) continue;
			if (shared!=memoryExpression.getrMemoryArrayAccess()) continue;
			if (!sharedMemories.contains(shared.getArrayName().toString())) continue;
			if (memoryExpression.getLHS() instanceof Identifier) {
				variables.add(memoryExpression.getLHS().toString());
			}
		}		
		
		List<String> iterators = new ArrayList();
		for (GLoop loop: func.getLoops()) {
			iterators.add(loop.getIterator().toString());
		}					
		
		dfi = new DepthFirstIterator(func.getProcedure());
		List<VariableDeclaration> decs = dfi.getList(VariableDeclaration.class);
		for (VariableDeclaration dec: decs) {
			VariableDeclaration sd = dec;
			if (DeclarationUtil.getVariableName(sd)!=null&&DeclarationUtil.getGSpecifier(sd)!=null) {
				String name = DeclarationUtil.getVariableName(sd);
				if (sharedMemories.contains(name)) continue;
				if (iterators.contains(name)) continue;
				if (variables.contains(name)) continue;
				decsTodo.add(sd);
			}
		}
		
		
		dfi = new DepthFirstIterator(func.getProcedure());
		List<ExpressionStatement> ess = (List<ExpressionStatement>)dfi.getList(ExpressionStatement.class);
		for (ExpressionStatement es: ess) {
			if (memoryStatments.contains(es.getExpression().toString())) continue;
			List<String> list = StatementUtil.parseVariableList(es.getExpression());
			list = StringUtil.subtract(list, iterators);
			list = StringUtil.subtract(list, sharedMemories);
			if (list.size()==0) continue;
			essTodo.add(es);
		}
		
	
	}

	/**
	 * when we duplicate, we can change tmp to tmp[8] to duplicate 8 copies
	 * @param expression
	 * @param decsTodo
	 * @param maxNumber
	 * @param isX
	 * @param oldid
	 * @param newid
	 * @param inc
	 * @param it
	 * @return
	 */
	static Expression updateAsArray(Expression expression,
				List<VariableDeclaration> decsTodo, int maxNumber, boolean isX,
				Identifier oldid, List<Expression> newid, int inc, Identifier it) {
			if (expression instanceof BinaryExpression) {
				BinaryExpression be = (BinaryExpression) expression;
	//			System.out.println(be);
				Expression lex = updateAsArray(be.getLHS(), decsTodo, maxNumber,
						isX, oldid, newid, inc, it);
				Expression rex = updateAsArray(be.getRHS(), decsTodo, maxNumber,
						isX, oldid, newid, inc, it);
				BinaryExpression nbe = new BinaryExpression(lex, be
						.getOperator(), rex);			
				return nbe;
			} else if (expression instanceof FunctionCall) {
				FunctionCall fc = (FunctionCall) expression;
				ArrayList<Expression> argus = new ArrayList();
				for (Object obj : fc.getArguments()) {
					Expression argu = (Expression) obj;
					Expression ex = updateAsArray(argu, decsTodo, maxNumber, isX,
							oldid, newid, inc, it);
					argus.add(ex);
				}
				FunctionCall nfc = new FunctionCall((Expression) fc.getName()
						.clone(), argus);
				return nfc;
				
			} else if (expression instanceof ArrayAccess) {
				ArrayAccess aa = (ArrayAccess) expression;
				// System.out.println("update:"+aa);
				Expression name = aa.getArrayName();
				boolean find = false;
				for (VariableDeclaration sd : decsTodo) {
					if (name.toString().equals(DeclarationUtil.getVariableName(sd).toString())) {
						find = true;
					}
				}
				if (!find) {
					Expression nname = updateAsArray(name, decsTodo, maxNumber, isX,
							oldid, newid, inc, it);
					List<Expression> indexs = new ArrayList();
					for (int i = 0; i < aa.getNumIndices(); i++) {
						Expression index = aa.getIndex(i);
						Expression exs = updateAsArray(index, decsTodo, maxNumber,
								isX, oldid, newid, inc, it);
						indexs.add(exs);
					}
					ArrayAccess newaa = new ArrayAccess(nname, indexs);
					return newaa;
				}
	
				if (name.toString().startsWith(CetusUtil.SYMBOL_PREFIX_SHARED) && aa.getNumIndices() == 1) {
					// 1d share memory, convert it to 2d
					// System.out.println("shared statement:"+aa.getStatement());
					ArrayList list = new ArrayList();
					list.add(aa.getIndex(0).clone());
					list.add((Identifier)it.clone());
					ArrayAccess newaa = new ArrayAccess(aa.getArrayName(), list);
					return newaa;
				}
	
				{
					Expression nname = updateAsArray(name, decsTodo, maxNumber, isX,
							oldid, newid, inc, it);
					List<Expression> indexs = new ArrayList();
					for (int i = 0; i < aa.getNumIndices(); i++) {
						Expression index = aa.getIndex(i);
						Expression exs = updateAsArray(index, decsTodo, maxNumber,
								isX, oldid, newid, inc, it);
						indexs.add(exs);
					}
					ArrayAccess newaa = new ArrayAccess(nname, indexs);
					return newaa;
				}
	
			} else if (expression instanceof Identifier) {
				Identifier id = (Identifier) expression;
	//			// System.out.println("update:"+id);
	//			if (oldid != null && oldid.toString().equals(id.toString())) {
	//				// System.out.println("find loop iterator:"+oldid);
	//				for (int i = 0; i < maxNumber; i++) {
	//					expressions.add((Expression) newid.get(i).clone());
	//				}
	//				return expressions;
	//			}
	
				if ((!isX && id.toString().equals(ThreadIndex.IDY))
						|| (isX && id.toString().equals(ThreadIndex.IDX))) {
					Expression ex = id;
					if (inc == 0) {
						ex = new BinaryExpression(ex, BinaryOperator.MULTIPLY,
								new IntegerLiteral(maxNumber));
						Expression newex = (Expression) ex.clone();
						newex = new BinaryExpression(ex, BinaryOperator.ADD,
								(Identifier)it.clone());
						return newex;
					} else {
						Expression newex = (Expression) ex.clone();
						newex = new BinaryExpression(ex, BinaryOperator.ADD,
								new BinaryExpression(new IntegerLiteral(inc), BinaryOperator.MULTIPLY, (Identifier)it.clone()));
						return newex;
					}
				}
	
				for (VariableDeclaration sd : decsTodo) {
					if (id.toString().equals(DeclarationUtil.getVariableName(sd).toString())) {
						ArrayAccess aa = new ArrayAccess((Identifier)id.clone(), (Identifier)it.clone());
						return aa;
					}
				}
				return (Identifier) id.clone();
			} else if (expression instanceof IntegerLiteral) {
				return (Expression) expression.clone();
			} 
			else if (expression instanceof UnaryExpression) {
				UnaryExpression ue = (UnaryExpression) expression;
				Expression exs = updateAsArray(ue.getExpression(), decsTodo, maxNumber,
						isX, oldid, newid, inc, it);
				return new UnaryExpression(ue.getOperator(), exs);
			}
			else {
				System.out.println("unsupport type" + expression + ";"
						+ expression.getClass());
			}
			return null;
		}

	/**
	 * when we duplicate, we can change tmp to tmp_0, tmp_1, tmp_2, .. tmp_7 to duplicate 8 copies
	 * and replace oldid to newid
	 * @param expression
	 * @param decsTodo
	 * @param maxNumber
	 * @param isX
	 * @param oldid
	 * @param newid
	 * @param inc
	 * @return
	 */
	static List<Expression> update(Expression expression,
			List<VariableDeclaration> decsTodo, int maxNumber, boolean isX,
			Identifier oldid, List<Expression> newid, int inc) {
		List<Expression> expressions = new ArrayList();
		if (expression instanceof BinaryExpression) {
			BinaryExpression be = (BinaryExpression) expression;
//			System.out.println(be);
			List<Expression> lex = update(be.getLHS(), decsTodo, maxNumber,
					isX, oldid, newid, inc);
			List<Expression> rex = update(be.getRHS(), decsTodo, maxNumber,
					isX, oldid, newid, inc);
			for (int i = 0; i < maxNumber; i++) {
				BinaryExpression nbe = new BinaryExpression(lex.get(i), be
						.getOperator(), rex.get(i));
				expressions.add(nbe);
			}

		} else if (expression instanceof FunctionCall) {
			FunctionCall fc = (FunctionCall) expression;
			ArrayList<List<Expression>> argus = new ArrayList();
			for (int i = 0; i < maxNumber; i++) {
				argus.add(new ArrayList());
			}
			for (Object obj : fc.getArguments()) {
				Expression argu = (Expression) obj;
				List<Expression> list = update(argu, decsTodo, maxNumber, isX,
						oldid, newid, inc);
				for (int i = 0; i < maxNumber; i++) {
					argus.get(i).add(list.get(i));
				}
			}
			for (int i = 0; i < maxNumber; i++) {
				FunctionCall nfc = new FunctionCall((Expression) fc.getName()
						.clone(), argus.get(i));
				expressions.add(nfc);
			}

		} else if (expression instanceof ArrayAccess) {
			ArrayAccess aa = (ArrayAccess) expression;
			// System.out.println("update:"+aa);
			Expression name = aa.getArrayName();
			boolean find = false;
			for (VariableDeclaration sd : decsTodo) {
				if (name.toString().equals(DeclarationUtil.getVariableName(sd))) {
					find = true;
				}
			}
			if (!find) {
				List<Expression> names = update(name, decsTodo, maxNumber, isX,
						oldid, newid, inc);
				List<List<Expression>> indexs = new ArrayList();
				for (int i = 0; i < maxNumber; i++) {
					indexs.add(new ArrayList());
				}
				for (int i = 0; i < aa.getNumIndices(); i++) {
					Expression index = aa.getIndex(i);
					List<Expression> exs = update(index, decsTodo, maxNumber,
							isX, oldid, newid, inc);
					for (int j = 0; j < maxNumber; j++) {
						indexs.get(j).add(exs.get(j));
					}
				}
				for (int i = 0; i < maxNumber; i++) {
					ArrayAccess newaa = new ArrayAccess(names.get(i), indexs
							.get(i));
					expressions.add(newaa);
				}
				return expressions;
			}

			if (name.toString().startsWith(CetusUtil.SYMBOL_PREFIX_SHARED) && aa.getNumIndices() == 1) {
				// 1d share memory, convert it to 2d
				// System.out.println("shared statement:"+aa.getStatement());
				for (int i = 0; i < maxNumber; i++) {
					ArrayList list = new ArrayList();
					list.add(aa.getIndex(0).clone());
					list.add(new IntegerLiteral(i));
					ArrayAccess newaa = new ArrayAccess(aa.getArrayName(), list);
					expressions.add(newaa);
				}
				return expressions;
			}

			{
				List<Expression> names = update(name, decsTodo, maxNumber, isX,
						oldid, newid, inc);
				List<List<Expression>> indexs = new ArrayList();
				for (int i = 0; i < maxNumber; i++) {
					indexs.add(new ArrayList());
				}
				for (int i = 0; i < aa.getNumIndices(); i++) {
					Expression index = aa.getIndex(i);
					List<Expression> exs = update(index, decsTodo, maxNumber,
							isX, oldid, newid, inc);
					for (int j = 0; j < maxNumber; j++) {
						indexs.get(j).add(exs.get(j));
					}
				}
				for (int i = 0; i < maxNumber; i++) {
					ArrayAccess newaa = new ArrayAccess(names.get(i), indexs
							.get(i));
					expressions.add(newaa);
				}
			}

		} else if (expression instanceof Identifier) {
			Identifier id = (Identifier) expression;
			// System.out.println("update:"+id);
			if (oldid != null && oldid.toString().equals(id.toString())) {
				// System.out.println("find loop iterator:"+oldid);
				for (int i = 0; i < maxNumber; i++) {
					expressions.add((Expression) newid.get(i).clone());
				}
				return expressions;
			}

			if ((!isX && id.toString().equals(ThreadIndex.IDY))
					|| (isX && id.toString().equals(ThreadIndex.IDX))) {
				Expression ex = id;
				if (inc == 0) {
					ex = new BinaryExpression(ex, BinaryOperator.MULTIPLY,
							new IntegerLiteral(maxNumber));
					for (int i = 0; i < maxNumber; i++) {
						Expression newex = (Expression) ex.clone();
						newex = new BinaryExpression(ex, BinaryOperator.ADD,
								new IntegerLiteral(i));
						expressions.add(newex);
					}
				} else {
					for (int i = 0; i < maxNumber; i++) {
						Expression newex = (Expression) ex.clone();
						newex = new BinaryExpression(ex, BinaryOperator.ADD,
								new IntegerLiteral(i * inc));
						expressions.add(newex);
					}
				}
				return expressions;
			}

			boolean find = false;
			for (VariableDeclaration sd : decsTodo) {
				if (id.toString().equals(DeclarationUtil.getVariableName(sd))) {
					for (int i = 0; i < maxNumber; i++) {
						Identifier ex = new Identifier(id.toString() + "_" + i);
						expressions.add(ex);
						find = true;
					}
				}
			}
			if (!find) {
				for (int i = 0; i < maxNumber; i++) {
					expressions.add((Identifier) id.clone());
				}
			}
		} else if (expression instanceof IntegerLiteral) {
			for (int i = 0; i < maxNumber; i++) {
				expressions.add((Expression) expression.clone());
			}
		} else if (expression instanceof UnaryExpression) {
			UnaryExpression ue = (UnaryExpression) expression;
			List<Expression> exs = update(ue.getExpression(), decsTodo, maxNumber,
					isX, oldid, newid, inc);
			
			for (int i = 0; i < maxNumber; i++) {
				expressions.add(new UnaryExpression(ue.getOperator(), exs.get(i)));
			}			
		} else {
			System.err.println("unsupport type" + expression + ";"
					+ expression.getClass());
		}
		return expressions;
	}
}
