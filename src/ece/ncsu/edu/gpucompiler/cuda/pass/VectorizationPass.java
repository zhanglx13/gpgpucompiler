package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import cetus.hir.AccessExpression;
import cetus.hir.AccessOperator;
import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.PointerSpecifier;
import cetus.hir.Specifier;
import cetus.hir.Tools;
import cetus.hir.Typecast;
import cetus.hir.UserSpecifier;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArray;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArrayAccess;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.VariableTools;
import ece.ncsu.edu.gpucompiler.cuda.index.Address;
import ece.ncsu.edu.gpucompiler.cuda.index.ConstIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.Index;

public class VectorizationPass extends Pass {

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}

	@Override
	public void dopass(GProcedure proc) {
		globalFilter(proc);
	}	
	
	void handleGlobal(List<MemoryArrayAccess> mas) {
		if (mas.size()==2) {
			MemoryArrayAccess ma0 = mas.get(0);
			MemoryArrayAccess ma1 = mas.get(1);
			if (ma0.getNumIndices()!=ma1.getNumIndices()) return;
			GProcedure proc = ma0.getMemoryExpression().getLoop().getGProcedure();
			Address x = ma0.getX().subtract(ma1.getX());
			Address y = new Address();
			if (ma0.getY()!=null&&ma1.getY()!=null) {
				y = ma0.getY().subtract(ma1.getY());
			}
			System.out.println(x.toExpression());
			System.out.println(y.toExpression());
			if (y.getIndexs().size()==0&&x.getIndexs().size()==1) {
				Index index = x.getIndexs().get(0);
				if (index instanceof ConstIndex) {
					ConstIndex ci = (ConstIndex)index;
					if (ci.getCoefficient()==1) {
						String name = ma0.getArrayName().toString();
//						System.out.println("find out neighbor global memory: "+name);
						
						MemoryArray stream = proc.getMemoryArray(name);
						if (stream.getType() == Specifier.FLOAT) {
//							System.out.println("the data type is float");
							MemoryArrayAccess small = ma1;
							if (ci.getCoefficient()<0) {
								small = ma0;
							}
							Address addr = small.getX();
							Address halfaddr = new Address();
							for (Index ind: addr.getIndexs()) {
								if (ind.getCoefficient()%2==1) {
									throw new RuntimeException("we don't support the odd address for: "+small);
								}
								Index nind;
								nind = (Index)ind.clone();
								nind.setCoefficient(nind.getCoefficient()/2);
								halfaddr.getIndexs().add(nind);

							}
							List<Specifier> specs = new ArrayList();
							specs.add(new UserSpecifier(new Identifier("struct")));
							specs.add(new UserSpecifier(new Identifier("float2")));
							specs.add(PointerSpecifier.UNQUALIFIED);
							Identifier global_f2 = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_TMP, proc.getProcedure());
							DeclarationStatement global_f2_init = StatementUtil.loadInitStatment(global_f2, specs);
							StatementUtil.addSibling(ma0.getStatement(), global_f2_init, true);
							Tools.addSymbols(proc.getProcedure(), global_f2_init.getDeclaration());
							Typecast tc = new Typecast(specs, new Identifier(name));
							ExpressionStatement convert = new ExpressionStatement(new AssignmentExpression(global_f2, AssignmentOperator.NORMAL, tc));
							StatementUtil.addSibling(global_f2_init, convert, false);
//							System.out.println(ae);
							specs = new ArrayList();
							specs.add(new UserSpecifier(new Identifier("struct")));
							specs.add(new UserSpecifier(new Identifier("float2")));							
							Identifier vid = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_TMP, proc.getProcedure());
							DeclarationStatement vid_init = StatementUtil.loadInitStatment(vid, specs);
							StatementUtil.addSibling(convert, vid_init, false);
							Tools.addSymbols(proc.getProcedure(), vid_init.getDeclaration());
							ArrayAccess aa = new ArrayAccess(global_f2, halfaddr.toExpression());
							
							ExpressionStatement vid_es = new ExpressionStatement(new AssignmentExpression(vid, AssignmentOperator.NORMAL, aa));
							StatementUtil.addSibling(vid_init, vid_es, false);
//							System.out.println(ae_vid);
							
							ma0.swapWith(new AccessExpression(vid, AccessOperator.MEMBER_ACCESS, new Identifier("x")));
							ma1.swapWith(new AccessExpression(vid, AccessOperator.MEMBER_ACCESS, new Identifier("y")));
							//CetusUtil.replace(ma0, new AccessExpression(vid, AccessOperator.MEMBER_ACCESS, new Identifier("x")));
							//CetusUtil.replace(ma1, new AccessExpression(vid, AccessOperator.MEMBER_ACCESS, new Identifier("y")));
							//System.out.println(proc.getProcedure());
							
							
						}
					}
				}
				
			}
		}
		
	}
	
	void globalFilter(GProcedure procedure) {
		DepthFirstIterator dfi = new DepthFirstIterator(procedure.getProcedure());
		List<GLoop> loops = dfi.getList(GLoop.class);		
		
		for (GLoop loop: loops) {
			boolean isHanlded = false;
			DepthFirstIterator dfi_me = new DepthFirstIterator(loop);
			List<MemoryExpression> mss = dfi_me.getList(MemoryExpression.class);		
	
			Hashtable<String, List<MemoryArrayAccess>> leftTable = new Hashtable();
			Hashtable<String, List<MemoryArrayAccess>> rightTable = new Hashtable();
			for (MemoryExpression ms: mss) {
				MemoryArrayAccess left = ms.getlMemoryArrayAccess();
				MemoryArrayAccess right = ms.getrMemoryArrayAccess();
				if (left!=null) {
					String name = left.getArrayName().toString();
					List<MemoryArrayAccess> mas = leftTable.get(name);
					if (mas==null) {
						mas = new ArrayList<MemoryArrayAccess>();
						leftTable.put(name, mas);
					}
					mas.add(left);
				}
				if (right!=null) {
					String name = right.getArrayName().toString();
					List<MemoryArrayAccess> mas = rightTable.get(name);
					if (mas==null) {
						mas = new ArrayList<MemoryArrayAccess>();
						rightTable.put(name, mas);
					}
					mas.add(right);
				}
			}
			
			for (List<MemoryArrayAccess> mas: leftTable.values()) {
				handleGlobal(mas);
				isHanlded = true;
			}
			for (List<MemoryArrayAccess> mas: rightTable.values()) {
				handleGlobal(mas);
				isHanlded = true;
			}
			if (isHanlded) {
				String nid = "";
				procedure.gerenateOutput(procedure.getProcedure().getName().toString()+"_"+getName()+"_"+nid);
			}
		}		

	}


}
